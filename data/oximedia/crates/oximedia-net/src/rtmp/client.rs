//! RTMP client implementation.
//!
//! This module provides a fully-featured RTMP client with support for:
//! - Connection handshake
//! - Publishing streams
//! - Playing streams
//! - Message sending and receiving
//! - Async I/O with tokio

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::single_match_else)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::format_collect)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unused_async)]
#![allow(clippy::identity_op)]

use super::{
    amf::{AmfDecoder, AmfEncoder, AmfValue},
    chunk::{AssembledMessage, ChunkStream, MessageHeader},
    handshake::{Handshake, HANDSHAKE_SIZE},
    message::{CommandMessage, ControlMessage, DataMessage, MessageType, RtmpMessage},
};
use crate::error::{NetError, NetResult};
use bytes::{Bytes, BytesMut};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

/// Default RTMP port.
pub const DEFAULT_RTMP_PORT: u16 = 1935;

/// Default connection timeout.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default chunk size.
pub const DEFAULT_CHUNK_SIZE: u32 = 4096;

/// Default window acknowledgement size.
pub const DEFAULT_WINDOW_ACK_SIZE: u32 = 2_500_000;

/// Chunk stream IDs.
pub mod chunk_stream_id {
    /// Protocol control messages.
    pub const PROTOCOL_CONTROL: u32 = 2;
    /// Command messages.
    pub const COMMAND: u32 = 3;
    /// Audio data.
    pub const AUDIO: u32 = 4;
    /// Video data.
    pub const VIDEO: u32 = 5;
    /// Data messages.
    pub const DATA: u32 = 6;
}

/// RTMP connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Not connected.
    Disconnected,
    /// Performing handshake.
    Handshaking,
    /// Connected, not initialized.
    Connected,
    /// Sent connect command, waiting for response.
    Connecting,
    /// Fully connected and ready.
    Ready,
    /// Publishing a stream.
    Publishing,
    /// Playing a stream.
    Playing,
    /// Error state.
    Error,
}

impl ConnectionState {
    /// Returns true if the connection is ready for operations.
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        matches!(self, Self::Ready | Self::Publishing | Self::Playing)
    }

    /// Returns true if connected (any state after handshake).
    #[must_use]
    pub const fn is_connected(&self) -> bool {
        !matches!(self, Self::Disconnected | Self::Error)
    }
}

/// RTMP URL components.
#[derive(Debug, Clone)]
pub struct RtmpUrl {
    /// Protocol (rtmp or rtmps).
    pub protocol: String,
    /// Host/IP address.
    pub host: String,
    /// Port number.
    pub port: u16,
    /// Application name.
    pub app: String,
    /// Stream name.
    pub stream: String,
}

impl RtmpUrl {
    /// Parses an RTMP URL.
    ///
    /// Format: `rtmp://host[:port]/app/stream`
    ///
    /// # Errors
    ///
    /// Returns an error if the URL is invalid.
    pub fn parse(url: &str) -> NetResult<Self> {
        let url = url.trim();

        // Check protocol
        let (protocol, rest) = if let Some(rest) = url.strip_prefix("rtmp://") {
            ("rtmp", rest)
        } else if let Some(rest) = url.strip_prefix("rtmps://") {
            ("rtmps", rest)
        } else {
            return Err(NetError::invalid_url(
                "URL must start with rtmp:// or rtmps://",
            ));
        };

        // Split into host:port and path
        let parts: Vec<&str> = rest.splitn(2, '/').collect();
        if parts.len() < 2 {
            return Err(NetError::invalid_url("URL must contain application path"));
        }

        let host_port = parts[0];
        let path = parts[1];

        // Parse host and port
        let (host, port) = if let Some((h, p)) = host_port.split_once(':') {
            let port = p
                .parse::<u16>()
                .map_err(|_| NetError::invalid_url("Invalid port number"))?;
            (h.to_string(), port)
        } else {
            (host_port.to_string(), DEFAULT_RTMP_PORT)
        };

        // Split path into app and stream
        let path_parts: Vec<&str> = path.splitn(2, '/').collect();
        let app = path_parts[0].to_string();
        let stream = if path_parts.len() > 1 {
            path_parts[1].to_string()
        } else {
            String::new()
        };

        Ok(Self {
            protocol: protocol.to_string(),
            host,
            port,
            app,
            stream,
        })
    }

    /// Returns the TCP connection address.
    #[must_use]
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Returns the tcUrl for connect command.
    #[must_use]
    pub fn tc_url(&self) -> String {
        format!("{}://{}/{}", self.protocol, self.host, self.app)
    }
}

/// RTMP client configuration.
#[derive(Debug, Clone)]
pub struct RtmpClientConfig {
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Read timeout.
    pub read_timeout: Duration,
    /// Write timeout.
    pub write_timeout: Duration,
    /// Initial chunk size.
    pub chunk_size: u32,
    /// Window acknowledgement size.
    pub window_ack_size: u32,
    /// Enable extended timestamp.
    pub extended_timestamp: bool,
}

impl Default for RtmpClientConfig {
    fn default() -> Self {
        Self {
            connect_timeout: DEFAULT_TIMEOUT,
            read_timeout: DEFAULT_TIMEOUT,
            write_timeout: DEFAULT_TIMEOUT,
            chunk_size: DEFAULT_CHUNK_SIZE,
            window_ack_size: DEFAULT_WINDOW_ACK_SIZE,
            extended_timestamp: true,
        }
    }
}

/// RTMP client session information.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    /// Current transaction ID counter.
    pub transaction_id: f64,
    /// Current stream ID.
    pub stream_id: u32,
    /// Server bandwidth.
    pub server_bandwidth: Option<u32>,
    /// Client bandwidth.
    pub client_bandwidth: Option<u32>,
    /// Bytes sent.
    pub bytes_sent: u64,
    /// Bytes received.
    pub bytes_received: u64,
    /// Connection timestamp.
    pub timestamp: u32,
}

impl SessionInfo {
    fn new() -> Self {
        Self {
            transaction_id: 1.0,
            stream_id: 0,
            server_bandwidth: None,
            client_bandwidth: None,
            bytes_sent: 0,
            bytes_received: 0,
            timestamp: 0,
        }
    }

    fn next_transaction_id(&mut self) -> f64 {
        let id = self.transaction_id;
        self.transaction_id += 1.0;
        id
    }

    fn update_timestamp(&mut self) {
        if let Ok(duration) = SystemTime::now().duration_since(UNIX_EPOCH) {
            self.timestamp = duration.as_millis() as u32;
        }
    }
}

/// Response callback type.
type ResponseCallback = Box<dyn FnOnce(NetResult<RtmpMessage>) + Send>;

/// RTMP client.
pub struct RtmpClient {
    /// TCP connection.
    stream: Option<TcpStream>,
    /// Configuration.
    config: RtmpClientConfig,
    /// Connection state.
    state: ConnectionState,
    /// Handshake handler.
    handshake: Handshake,
    /// Chunk stream handler.
    chunk_stream: ChunkStream,
    /// Session information.
    session: SessionInfo,
    /// URL information.
    url: Option<RtmpUrl>,
    /// Pending responses indexed by transaction ID.
    pending_responses: HashMap<u32, ResponseCallback>,
    /// Read buffer.
    read_buffer: BytesMut,
    /// Current chunk size.
    chunk_size: u32,
}

impl RtmpClient {
    /// Creates a new RTMP client.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(RtmpClientConfig::default())
    }

    /// Creates a new RTMP client with custom configuration.
    #[must_use]
    pub fn with_config(config: RtmpClientConfig) -> Self {
        let chunk_size = config.chunk_size;
        let mut chunk_stream = ChunkStream::new();
        chunk_stream.set_tx_chunk_size(chunk_size);
        chunk_stream.set_rx_chunk_size(chunk_size);

        Self {
            stream: None,
            config,
            state: ConnectionState::Disconnected,
            handshake: Handshake::new(),
            chunk_stream,
            session: SessionInfo::new(),
            url: None,
            pending_responses: HashMap::new(),
            read_buffer: BytesMut::with_capacity(8192),
            chunk_size,
        }
    }

    /// Returns the current connection state.
    #[must_use]
    pub const fn state(&self) -> ConnectionState {
        self.state
    }

    /// Returns true if connected.
    #[must_use]
    pub const fn is_connected(&self) -> bool {
        self.state.is_connected()
    }

    /// Returns true if ready for operations.
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        self.state.is_ready()
    }

    /// Returns session information.
    #[must_use]
    pub const fn session_info(&self) -> &SessionInfo {
        &self.session
    }

    /// Connects to an RTMP server.
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails.
    pub async fn connect(&mut self, url: &str) -> NetResult<()> {
        // Parse URL
        let rtmp_url = RtmpUrl::parse(url)?;
        let address = rtmp_url.address();
        self.url = Some(rtmp_url.clone());

        // Connect TCP
        self.state = ConnectionState::Handshaking;
        let stream = timeout(self.config.connect_timeout, TcpStream::connect(&address))
            .await
            .map_err(|_| NetError::timeout(format!("Connection to {address} timed out")))?
            .map_err(|e| NetError::connection(format!("Failed to connect to {address}: {e}")))?;

        self.stream = Some(stream);

        // Perform handshake
        self.perform_handshake().await?;

        self.state = ConnectionState::Connected;

        // Send connect command
        self.send_connect_command(&rtmp_url).await?;

        Ok(())
    }

    /// Performs RTMP handshake.
    async fn perform_handshake(&mut self) -> NetResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| NetError::invalid_state("No active connection"))?;

        // Update timestamp
        self.session.update_timestamp();
        self.handshake.set_epoch(self.session.timestamp);

        // Send C0+C1
        let c0c1 = self.handshake.generate_c0c1();
        timeout(self.config.write_timeout, stream.write_all(&c0c1))
            .await
            .map_err(|_| NetError::timeout("Handshake write timeout"))?
            .map_err(|e| NetError::handshake(format!("Failed to send C0+C1: {e}")))?;

        self.session.bytes_sent += c0c1.len() as u64;

        // Read S0+S1+S2 (1 + 1536 + 1536 = 3073 bytes)
        let mut buf = vec![0u8; 1 + HANDSHAKE_SIZE * 2];
        timeout(self.config.read_timeout, stream.read_exact(&mut buf))
            .await
            .map_err(|_| NetError::timeout("Handshake read timeout"))?
            .map_err(|e| NetError::handshake(format!("Failed to read S0+S1+S2: {e}")))?;

        self.session.bytes_received += buf.len() as u64;

        // Parse S0+S1 (captures the server's S1 timestamp + random data,
        // needed below to build C2's echo payload).
        self.handshake.parse_s0s1(&buf[..1 + HANDSHAKE_SIZE])?;

        // Simple RTMP handshake, client role, in protocol order:
        //   1. send C0+C1                (generate_c0c1 -> VersionSent)
        //   2. receive S0+S1+S2          (parse_s0s1, above)
        //   3. send C2 (echo of S1)      (generate_c2   -> AckSent)
        //   4. validate S2 (echo of C1)  (parse_s2      -> Done)
        //
        // Step 3 MUST run before step 4: `generate_c2` marks the
        // intermediate `AckSent` state and `parse_s2` marks the terminal
        // `Done` state. Calling `parse_s2` first and `generate_c2` second
        // (the previous, buggy order) clobbered `Done` back down to
        // `AckSent` after a fully successful handshake, so `is_done()`
        // permanently returned false and every client handshake failed
        // with "Handshake incomplete" even though both sides had
        // completed the exchange correctly.
        let c2 = self.handshake.generate_c2();
        timeout(self.config.write_timeout, stream.write_all(&c2))
            .await
            .map_err(|_| NetError::timeout("Handshake C2 write timeout"))?
            .map_err(|e| NetError::handshake(format!("Failed to send C2: {e}")))?;

        self.session.bytes_sent += c2.len() as u64;

        // Validate S2 (echo of our C1) - this is the final handshake step
        // and transitions the state machine to `Done`.
        self.handshake.parse_s2(&buf[1 + HANDSHAKE_SIZE..])?;

        // Verify handshake is done
        if !self.handshake.is_done() {
            return Err(NetError::handshake("Handshake incomplete"));
        }

        Ok(())
    }

    /// Sends the connect command.
    async fn send_connect_command(&mut self, url: &RtmpUrl) -> NetResult<()> {
        self.state = ConnectionState::Connecting;

        // Send connect command
        let cmd = CommandMessage::connect(&url.app, &url.tc_url());
        self.send_command(cmd, chunk_stream_id::COMMAND).await?;

        // Wait for connect response
        self.wait_for_connect_response().await?;

        self.state = ConnectionState::Ready;
        Ok(())
    }

    /// Waits for connect response.
    async fn wait_for_connect_response(&mut self) -> NetResult<()> {
        loop {
            let messages = self.read_messages().await?;

            for msg in messages {
                if let RtmpMessage::Command(cmd) = msg {
                    if cmd.name == "_result" || cmd.name == "onBWDone" {
                        return Ok(());
                    }
                    if cmd.name == "_error" {
                        return Err(NetError::protocol("Connect command failed"));
                    }
                }
            }
        }
    }

    /// Publishes a stream.
    ///
    /// # Errors
    ///
    /// Returns an error if publish fails.
    pub async fn publish(&mut self, stream_name: &str, publish_type: &str) -> NetResult<()> {
        if !self.is_ready() {
            return Err(NetError::invalid_state("Client not ready for publishing"));
        }

        // Create stream
        let stream_id = self.create_stream().await?;
        self.session.stream_id = stream_id;

        // Send publish command
        let transaction_id = self.session.next_transaction_id();
        let cmd = CommandMessage::publish(stream_name, publish_type, transaction_id);
        self.send_command(cmd, chunk_stream_id::COMMAND).await?;

        // Wait for publish response
        self.wait_for_publish_response().await?;

        self.state = ConnectionState::Publishing;
        Ok(())
    }

    /// Waits for publish response.
    async fn wait_for_publish_response(&mut self) -> NetResult<()> {
        loop {
            let messages = self.read_messages().await?;

            for msg in messages {
                if let RtmpMessage::Command(cmd) = msg {
                    if cmd.name == "onStatus" {
                        // Check status
                        if let Some(info) = cmd.args.first() {
                            if let Some(obj) = info.as_object() {
                                if let Some(code) = obj.get("code") {
                                    if let Some(code_str) = code.as_str() {
                                        if code_str.contains("Success")
                                            || code_str.contains("Start")
                                        {
                                            return Ok(());
                                        }
                                        if code_str.contains("Error") || code_str.contains("Fail") {
                                            return Err(NetError::protocol(format!(
                                                "Publish failed: {code_str}"
                                            )));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Plays a stream.
    ///
    /// # Errors
    ///
    /// Returns an error if play fails.
    pub async fn play(&mut self, stream_name: &str) -> NetResult<()> {
        if !self.is_ready() {
            return Err(NetError::invalid_state("Client not ready for playing"));
        }

        // Create stream
        let stream_id = self.create_stream().await?;
        self.session.stream_id = stream_id;

        // Send play command
        let transaction_id = self.session.next_transaction_id();
        let cmd = CommandMessage::play(stream_name, transaction_id);
        self.send_command(cmd, chunk_stream_id::COMMAND).await?;

        // Wait for play response
        self.wait_for_play_response().await?;

        self.state = ConnectionState::Playing;
        Ok(())
    }

    /// Waits for play response.
    async fn wait_for_play_response(&mut self) -> NetResult<()> {
        loop {
            let messages = self.read_messages().await?;

            for msg in messages {
                if let RtmpMessage::Command(cmd) = msg {
                    if cmd.name == "onStatus" {
                        // Check status
                        if let Some(info) = cmd.args.first() {
                            if let Some(obj) = info.as_object() {
                                if let Some(code) = obj.get("code") {
                                    if let Some(code_str) = code.as_str() {
                                        if code_str.contains("Start") || code_str.contains("Reset")
                                        {
                                            return Ok(());
                                        }
                                        if code_str.contains("Error") || code_str.contains("Fail") {
                                            return Err(NetError::protocol(format!(
                                                "Play failed: {code_str}"
                                            )));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Creates a stream and returns the stream ID.
    async fn create_stream(&mut self) -> NetResult<u32> {
        let transaction_id = self.session.next_transaction_id();
        let cmd = CommandMessage::create_stream(transaction_id);
        self.send_command(cmd, chunk_stream_id::COMMAND).await?;

        // Wait for createStream response
        loop {
            let messages = self.read_messages().await?;

            for msg in messages {
                if let RtmpMessage::Command(cmd) = msg {
                    if cmd.name == "_result" && !cmd.args.is_empty() {
                        if let Some(stream_id) = cmd.args[0].as_number() {
                            return Ok(stream_id as u32);
                        }
                    }
                }
            }
        }
    }

    /// Sends a command message.
    async fn send_command(&mut self, cmd: CommandMessage, csid: u32) -> NetResult<()> {
        let msg = RtmpMessage::Command(cmd);
        self.send_message(msg, csid).await
    }

    /// Sends a data message.
    pub async fn send_data(&mut self, data: DataMessage) -> NetResult<()> {
        let msg = RtmpMessage::Data(data);
        self.send_message(msg, chunk_stream_id::DATA).await
    }

    /// Sends an audio packet.
    pub async fn send_audio(&mut self, data: Bytes) -> NetResult<()> {
        let msg = RtmpMessage::Audio(data);
        self.send_message(msg, chunk_stream_id::AUDIO).await
    }

    /// Sends a video packet.
    pub async fn send_video(&mut self, data: Bytes) -> NetResult<()> {
        let msg = RtmpMessage::Video(data);
        self.send_message(msg, chunk_stream_id::VIDEO).await
    }

    /// Sends an RTMP message.
    pub async fn send_message(&mut self, message: RtmpMessage, csid: u32) -> NetResult<()> {
        // Encode message payload
        let payload = self.encode_message_payload(&message)?;

        // Update timestamp
        self.session.update_timestamp();

        // Create message header
        let header = MessageHeader::new(
            self.session.timestamp,
            payload.len() as u32,
            message.type_id(),
            self.session.stream_id,
        );

        // Encode chunks
        let chunks = self.chunk_stream.encode_message(csid, &header, &payload);

        // Send chunks
        self.write_bytes(&chunks).await?;

        Ok(())
    }

    /// Encodes message payload.
    fn encode_message_payload(&self, message: &RtmpMessage) -> NetResult<Bytes> {
        match message {
            RtmpMessage::Control(ctrl) => Ok(ctrl.encode()),
            RtmpMessage::Command(cmd) => self.encode_command(cmd),
            RtmpMessage::Data(data) => self.encode_data(data),
            RtmpMessage::Audio(bytes) => Ok(bytes.clone()),
            RtmpMessage::Video(bytes) => Ok(bytes.clone()),
            RtmpMessage::Unknown { payload, .. } => Ok(payload.clone()),
        }
    }

    /// Encodes a command message.
    fn encode_command(&self, cmd: &CommandMessage) -> NetResult<Bytes> {
        let mut enc = AmfEncoder::new();

        // Command name
        enc.encode(&AmfValue::String(cmd.name.clone()));

        // Transaction ID
        enc.encode(&AmfValue::Number(cmd.transaction_id));

        // Command object
        if let Some(obj) = &cmd.command_object {
            enc.encode(obj);
        } else {
            enc.encode(&AmfValue::Null);
        }

        // Arguments
        for arg in &cmd.args {
            enc.encode(arg);
        }

        Ok(enc.finish())
    }

    /// Encodes a data message.
    fn encode_data(&self, data: &DataMessage) -> NetResult<Bytes> {
        let mut enc = AmfEncoder::new();

        // Handler name
        enc.encode(&AmfValue::String(data.handler.clone()));

        // Values
        for value in &data.values {
            enc.encode(value);
        }

        Ok(enc.finish())
    }

    /// Reads RTMP messages.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails.
    pub async fn read_messages(&mut self) -> NetResult<Vec<RtmpMessage>> {
        // Read data into buffer
        self.read_chunk_data().await?;

        // Process chunks
        let assembled = self.chunk_stream.process_chunk(&self.read_buffer)?;
        self.read_buffer.clear();

        // Decode messages
        let mut messages = Vec::new();
        for msg in assembled {
            let rtmp_msg = self.decode_message(msg)?;

            // Handle control messages
            if let RtmpMessage::Control(ref ctrl) = rtmp_msg {
                self.handle_control_message(ctrl).await?;
            }

            messages.push(rtmp_msg);
        }

        Ok(messages)
    }

    /// Reads chunk data into buffer.
    async fn read_chunk_data(&mut self) -> NetResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| NetError::invalid_state("No active connection"))?;

        // Read at least one chunk
        let mut temp_buf = vec![0u8; self.chunk_size as usize * 4];

        let n = timeout(self.config.read_timeout, stream.read(&mut temp_buf))
            .await
            .map_err(|_| NetError::timeout("Read timeout"))?
            .map_err(|e| NetError::connection(format!("Read failed: {e}")))?;

        if n == 0 {
            return Err(NetError::Eof);
        }

        self.read_buffer.extend_from_slice(&temp_buf[..n]);
        self.session.bytes_received += n as u64;

        Ok(())
    }

    /// Decodes an assembled message.
    fn decode_message(&self, msg: AssembledMessage) -> NetResult<RtmpMessage> {
        let msg_type = MessageType::from_id(msg.header.message_type).ok_or_else(|| {
            NetError::protocol(format!("Unknown message type: {}", msg.header.message_type))
        })?;

        if msg_type.is_control() {
            let ctrl = ControlMessage::decode(msg_type, &msg.payload)?;
            Ok(RtmpMessage::Control(ctrl))
        } else if msg_type.is_command() {
            self.decode_command(&msg.payload)
        } else if msg_type == MessageType::DataAmf0 {
            self.decode_data(&msg.payload)
        } else if msg_type == MessageType::Audio {
            Ok(RtmpMessage::Audio(msg.payload))
        } else if msg_type == MessageType::Video {
            Ok(RtmpMessage::Video(msg.payload))
        } else {
            Ok(RtmpMessage::Unknown {
                type_id: msg.header.message_type,
                payload: msg.payload,
            })
        }
    }

    /// Decodes a command message.
    fn decode_command(&self, data: &[u8]) -> NetResult<RtmpMessage> {
        let mut dec = AmfDecoder::new(data);

        // Command name
        let name = dec
            .decode()?
            .as_str()
            .ok_or_else(|| NetError::encoding("Command name must be string"))?
            .to_string();

        // Transaction ID
        let transaction_id = dec
            .decode()?
            .as_number()
            .ok_or_else(|| NetError::encoding("Transaction ID must be number"))?;

        // Command object
        let command_object = if dec.has_remaining() {
            Some(dec.decode()?)
        } else {
            None
        };

        // Arguments
        let mut args = Vec::new();
        while dec.has_remaining() {
            args.push(dec.decode()?);
        }

        Ok(RtmpMessage::Command(CommandMessage {
            name,
            transaction_id,
            command_object,
            args,
        }))
    }

    /// Decodes a data message.
    fn decode_data(&self, data: &[u8]) -> NetResult<RtmpMessage> {
        let mut dec = AmfDecoder::new(data);

        // Handler name
        let handler = dec
            .decode()?
            .as_str()
            .ok_or_else(|| NetError::encoding("Handler must be string"))?
            .to_string();

        // Values
        let mut values = Vec::new();
        while dec.has_remaining() {
            values.push(dec.decode()?);
        }

        Ok(RtmpMessage::Data(DataMessage { handler, values }))
    }

    /// Handles control messages.
    async fn handle_control_message(&mut self, ctrl: &ControlMessage) -> NetResult<()> {
        match ctrl {
            ControlMessage::SetChunkSize(size) => {
                self.chunk_stream.set_rx_chunk_size(*size);
                self.chunk_size = *size;
            }
            ControlMessage::WindowAckSize(size) => {
                self.session.server_bandwidth = Some(*size);
            }
            ControlMessage::SetPeerBandwidth { size, .. } => {
                self.session.client_bandwidth = Some(*size);
                // Send acknowledgement
                let ack = ControlMessage::WindowAckSize(*size);
                let msg = RtmpMessage::Control(ack);
                self.send_message(msg, chunk_stream_id::PROTOCOL_CONTROL)
                    .await?;
            }
            ControlMessage::UserControl { .. } => {
                // Handle user control events if needed
            }
            _ => {}
        }
        Ok(())
    }

    /// Writes bytes to the stream.
    async fn write_bytes(&mut self, data: &[u8]) -> NetResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| NetError::invalid_state("No active connection"))?;

        timeout(self.config.write_timeout, stream.write_all(data))
            .await
            .map_err(|_| NetError::timeout("Write timeout"))?
            .map_err(|e| NetError::connection(format!("Write failed: {e}")))?;

        self.session.bytes_sent += data.len() as u64;
        Ok(())
    }

    /// Closes the connection.
    pub async fn close(&mut self) -> NetResult<()> {
        if let Some(mut stream) = self.stream.take() {
            let _ = stream.shutdown().await;
        }
        self.state = ConnectionState::Disconnected;
        Ok(())
    }

    /// Sends metadata (onMetaData).
    pub async fn send_metadata(&mut self, metadata: HashMap<String, AmfValue>) -> NetResult<()> {
        let data = DataMessage::on_metadata(AmfValue::Object(metadata));
        self.send_data(data).await
    }

    /// Sets chunk size.
    pub async fn set_chunk_size(&mut self, size: u32) -> NetResult<()> {
        let ctrl = ControlMessage::SetChunkSize(size);
        let msg = RtmpMessage::Control(ctrl);
        self.send_message(msg, chunk_stream_id::PROTOCOL_CONTROL)
            .await?;

        self.chunk_stream.set_tx_chunk_size(size);
        self.chunk_size = size;
        Ok(())
    }

    /// Sends acknowledgement.
    pub async fn send_acknowledgement(&mut self, sequence: u32) -> NetResult<()> {
        let ctrl = ControlMessage::Acknowledgement(sequence);
        let msg = RtmpMessage::Control(ctrl);
        self.send_message(msg, chunk_stream_id::PROTOCOL_CONTROL)
            .await
    }

    /// Sends window acknowledgement size.
    pub async fn send_window_ack_size(&mut self, size: u32) -> NetResult<()> {
        let ctrl = ControlMessage::WindowAckSize(size);
        let msg = RtmpMessage::Control(ctrl);
        self.send_message(msg, chunk_stream_id::PROTOCOL_CONTROL)
            .await
    }

    /// Sends set peer bandwidth.
    pub async fn send_peer_bandwidth(&mut self, size: u32, limit_type: u8) -> NetResult<()> {
        let ctrl = ControlMessage::SetPeerBandwidth { size, limit_type };
        let msg = RtmpMessage::Control(ctrl);
        self.send_message(msg, chunk_stream_id::PROTOCOL_CONTROL)
            .await
    }

    /// Sends user control event.
    #[allow(clippy::too_many_arguments)]
    pub async fn send_user_control_event(
        &mut self,
        event: super::message::UserControlEvent,
        data: u32,
        extra: Option<u32>,
    ) -> NetResult<()> {
        let ctrl = ControlMessage::UserControl { event, data, extra };
        let msg = RtmpMessage::Control(ctrl);
        self.send_message(msg, chunk_stream_id::PROTOCOL_CONTROL)
            .await
    }

    /// Sends stream begin event.
    pub async fn send_stream_begin(&mut self, stream_id: u32) -> NetResult<()> {
        self.send_user_control_event(
            super::message::UserControlEvent::StreamBegin,
            stream_id,
            None,
        )
        .await
    }

    /// Sends ping request.
    pub async fn send_ping_request(&mut self, timestamp: u32) -> NetResult<()> {
        self.send_user_control_event(
            super::message::UserControlEvent::PingRequest,
            timestamp,
            None,
        )
        .await
    }

    /// Sends ping response.
    pub async fn send_ping_response(&mut self, timestamp: u32) -> NetResult<()> {
        self.send_user_control_event(
            super::message::UserControlEvent::PingResponse,
            timestamp,
            None,
        )
        .await
    }

    /// Receives the next message with timeout.
    pub async fn receive_message(&mut self) -> NetResult<RtmpMessage> {
        let messages = self.read_messages().await?;
        messages
            .into_iter()
            .next()
            .ok_or_else(|| NetError::protocol("No message received"))
    }

    /// Receives messages with a custom filter.
    pub async fn receive_messages_filtered<F>(
        &mut self,
        mut filter: F,
    ) -> NetResult<Vec<RtmpMessage>>
    where
        F: FnMut(&RtmpMessage) -> bool,
    {
        let messages = self.read_messages().await?;
        Ok(messages.into_iter().filter(|m| filter(m)).collect())
    }

    /// Flushes pending data.
    pub async fn flush(&mut self) -> NetResult<()> {
        if let Some(stream) = self.stream.as_mut() {
            timeout(self.config.write_timeout, stream.flush())
                .await
                .map_err(|_| NetError::timeout("Flush timeout"))?
                .map_err(|e| NetError::connection(format!("Flush failed: {e}")))?;
        }
        Ok(())
    }

    /// Returns the current URL.
    #[must_use]
    pub const fn url(&self) -> Option<&RtmpUrl> {
        self.url.as_ref()
    }

    /// Returns bytes sent.
    #[must_use]
    pub const fn bytes_sent(&self) -> u64 {
        self.session.bytes_sent
    }

    /// Returns bytes received.
    #[must_use]
    pub const fn bytes_received(&self) -> u64 {
        self.session.bytes_received
    }

    /// Returns the current stream ID.
    #[must_use]
    pub const fn stream_id(&self) -> u32 {
        self.session.stream_id
    }
}

impl Default for RtmpClient {
    fn default() -> Self {
        Self::new()
    }
}

/// RTMP client builder for easier configuration.
#[derive(Debug, Clone)]
pub struct RtmpClientBuilder {
    config: RtmpClientConfig,
}

impl RtmpClientBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RtmpClientConfig::default(),
        }
    }

    /// Sets the connection timeout.
    #[must_use]
    pub const fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.config.connect_timeout = timeout;
        self
    }

    /// Sets the read timeout.
    #[must_use]
    pub const fn read_timeout(mut self, timeout: Duration) -> Self {
        self.config.read_timeout = timeout;
        self
    }

    /// Sets the write timeout.
    #[must_use]
    pub const fn write_timeout(mut self, timeout: Duration) -> Self {
        self.config.write_timeout = timeout;
        self
    }

    /// Sets the chunk size.
    #[must_use]
    pub const fn chunk_size(mut self, size: u32) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Sets the window acknowledgement size.
    #[must_use]
    pub const fn window_ack_size(mut self, size: u32) -> Self {
        self.config.window_ack_size = size;
        self
    }

    /// Enables or disables extended timestamp.
    #[must_use]
    pub const fn extended_timestamp(mut self, enabled: bool) -> Self {
        self.config.extended_timestamp = enabled;
        self
    }

    /// Builds the RTMP client.
    #[must_use]
    pub fn build(self) -> RtmpClient {
        RtmpClient::with_config(self.config)
    }
}

impl Default for RtmpClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// High-level client types (stream key, state machine, stats, media packets)
// ---------------------------------------------------------------------------

/// RTMP stream key for publishing.
#[derive(Debug, Clone)]
pub struct StreamKey {
    /// Application name (e.g. "live").
    pub app: String,
    /// Stream key (e.g. "stream1").
    pub key: String,
}

impl StreamKey {
    /// Creates a new stream key.
    pub fn new(app: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            app: app.into(),
            key: key.into(),
        }
    }

    /// Parses a stream key from an RTMP URL.
    ///
    /// Accepts `rtmp://host/app/key` format.
    pub fn from_url(url: &str) -> Result<Self, String> {
        // Strip protocol prefix
        let rest = url
            .strip_prefix("rtmp://")
            .or_else(|| url.strip_prefix("rtmps://"))
            .ok_or_else(|| format!("URL must start with rtmp:// or rtmps://: {url}"))?;

        // Drop host (everything up to first '/')
        let path = rest
            .splitn(2, '/')
            .nth(1)
            .ok_or_else(|| format!("URL must contain application path: {url}"))?;

        // Split path into app/key
        let mut parts = path.splitn(2, '/');
        let app = parts
            .next()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("URL must contain app name: {url}"))?;
        let key = parts.next().unwrap_or("");

        Ok(Self::new(app, key))
    }
}

/// RTMP publish mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishMode {
    /// Live broadcast stream.
    Live,
    /// Record stream to file on server.
    Record,
    /// Append to existing recording.
    Append,
}

impl PublishMode {
    /// Returns the string representation used in RTMP commands.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Record => "record",
            Self::Append => "append",
        }
    }
}

/// Statistics for an active RTMP connection.
#[derive(Debug, Clone, Default)]
pub struct RtmpClientStats {
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Total packets sent.
    pub packets_sent: u64,
    /// Total packets received.
    pub packets_received: u64,
    /// Number of frames dropped.
    pub dropped_frames: u64,
    /// Connection duration in milliseconds.
    pub connection_duration_ms: u64,
}

/// RTMP client state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientState {
    /// Not connected.
    Disconnected,
    /// TCP connection in progress.
    Connecting,
    /// RTMP handshake in progress.
    Handshaking,
    /// Connected and application-level session established.
    Connected,
    /// Publishing a stream.
    Publishing,
    /// Playing a stream.
    Playing,
    /// Graceful shutdown in progress.
    Disconnecting,
    /// An unrecoverable error has occurred.
    Error,
}

impl ClientState {
    /// Returns `true` if the client is in an active streaming state.
    pub fn is_active(self) -> bool {
        matches!(self, Self::Connected | Self::Publishing | Self::Playing)
    }

    /// Returns `true` if the client is ready to start publishing.
    pub fn can_publish(self) -> bool {
        matches!(self, Self::Connected)
    }
}

/// Type of media packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtmpMediaPacketType {
    /// Video keyframe (IDR).
    VideoKeyframe,
    /// Video inter-frame.
    VideoInterframe,
    /// AAC audio (included for completeness; prefer Opus).
    AudioAAC,
    /// Opus audio (patent-free).
    AudioOpus,
    /// Stream metadata message.
    Metadata,
}

/// Media packet for RTMP publishing.
#[derive(Debug, Clone)]
pub struct RtmpMediaPacket {
    /// Packet type.
    pub packet_type: RtmpMediaPacketType,
    /// Presentation timestamp in milliseconds.
    pub timestamp_ms: u32,
    /// Raw payload data.
    pub data: Vec<u8>,
    /// Whether this is a keyframe.
    pub is_keyframe: bool,
    /// Composition time offset for video (B-frame support).
    pub composition_time_offset: i32,
}

/// Simplified RTMP client for building messages and tracking state
/// without requiring an active TCP connection.
pub struct SimpleRtmpClient {
    config: RtmpClientConfig,
    state: ClientState,
    stats: RtmpClientStats,
    stream_key: Option<StreamKey>,
    chunk_sequence_number: u32,
}

impl SimpleRtmpClient {
    /// Creates a new client with the given configuration.
    pub fn new(config: RtmpClientConfig) -> Self {
        Self {
            config,
            state: ClientState::Disconnected,
            stats: RtmpClientStats::default(),
            stream_key: None,
            chunk_sequence_number: 0,
        }
    }

    /// Creates a new client with the default configuration.
    pub fn with_default_config() -> Self {
        Self::new(RtmpClientConfig::default())
    }

    /// Returns the current connection state.
    pub fn state(&self) -> ClientState {
        self.state
    }

    /// Returns connection statistics.
    pub fn stats(&self) -> &RtmpClientStats {
        &self.stats
    }

    /// Returns the connected stream key, if any.
    pub fn stream_key(&self) -> Option<&StreamKey> {
        self.stream_key.as_ref()
    }

    /// Builds a connect command payload (AMF0 encoded).
    pub fn build_connect_message(&self, app: &str, tc_url: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        // Command name: "connect"
        buf.extend(amf0::encode_string("connect"));
        // Transaction ID: 1.0
        buf.extend(amf0::encode_number(1.0));
        // Command object
        buf.extend(amf0::start_object());
        buf.extend(amf0::encode_property("app", &amf0::encode_string(app)));
        buf.extend(amf0::encode_property("tcUrl", &amf0::encode_string(tc_url)));
        buf.extend(amf0::encode_property(
            "type",
            &amf0::encode_string("nonprivate"),
        ));
        buf.extend(amf0::end_object());
        buf
    }

    /// Builds a createStream command payload.
    pub fn build_create_stream_message(&self, transaction_id: f64) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend(amf0::encode_string("createStream"));
        buf.extend(amf0::encode_number(transaction_id));
        buf.extend(amf0::encode_null());
        buf
    }

    /// Builds a publish command payload.
    pub fn build_publish_message(
        &self,
        stream_id: u32,
        stream_name: &str,
        publish_type: PublishMode,
    ) -> Vec<u8> {
        let _ = stream_id; // stream_id used by caller to set RTMP stream ID
        let mut buf = Vec::new();
        buf.extend(amf0::encode_string("publish"));
        buf.extend(amf0::encode_number(0.0)); // transaction ID 0 for publish
        buf.extend(amf0::encode_null());
        buf.extend(amf0::encode_string(stream_name));
        buf.extend(amf0::encode_string(publish_type.as_str()));
        buf
    }

    /// Builds an RTMP video message payload.
    ///
    /// The payload begins with a 4-byte timestamp in network byte order
    /// followed by the raw frame data.
    pub fn build_video_message(&self, packet: &RtmpMediaPacket, stream_id: u32) -> Vec<u8> {
        let _ = stream_id;
        let mut buf = Vec::new();
        // Timestamp (big-endian)
        buf.extend_from_slice(&packet.timestamp_ms.to_be_bytes());
        // Keyframe flag byte (simple header)
        let flags: u8 = if packet.is_keyframe { 0x17 } else { 0x27 };
        buf.push(flags);
        // Composition time (3 bytes big-endian, lower 24 bits)
        let ct = packet.composition_time_offset;
        buf.push(((ct >> 16) & 0xFF) as u8);
        buf.push(((ct >> 8) & 0xFF) as u8);
        buf.push((ct & 0xFF) as u8);
        // Payload
        buf.extend_from_slice(&packet.data);
        buf
    }

    /// Builds an onMetaData message payload.
    pub fn build_metadata_message(
        &self,
        width: u32,
        height: u32,
        frame_rate: f64,
        stream_id: u32,
    ) -> Vec<u8> {
        let _ = stream_id;
        let mut buf = Vec::new();
        buf.extend(amf0::encode_string("@setDataFrame"));
        buf.extend(amf0::encode_string("onMetaData"));
        buf.extend(amf0::start_object());
        buf.extend(amf0::encode_property(
            "width",
            &amf0::encode_number(f64::from(width)),
        ));
        buf.extend(amf0::encode_property(
            "height",
            &amf0::encode_number(f64::from(height)),
        ));
        buf.extend(amf0::encode_property(
            "framerate",
            &amf0::encode_number(frame_rate),
        ));
        buf.extend(amf0::end_object());
        buf
    }

    /// Records bytes sent and increments packet counter.
    pub fn record_send(&mut self, bytes: u64) {
        self.stats.bytes_sent += bytes;
        self.stats.packets_sent += 1;
    }

    /// Records bytes received and increments packet counter.
    pub fn record_receive(&mut self, bytes: u64) {
        self.stats.bytes_received += bytes;
        self.stats.packets_received += 1;
    }

    /// Simulates a successful connect transition (for testing).
    pub fn simulate_connect(&mut self, stream_key: StreamKey) {
        self.state = ClientState::Connected;
        self.stream_key = Some(stream_key);
    }

    /// Simulates a successful publish transition (for testing).
    pub fn simulate_publish(&mut self) {
        if self.state == ClientState::Connected {
            self.state = ClientState::Publishing;
        }
    }

    /// Simulates a disconnect (for testing).
    pub fn simulate_disconnect(&mut self) {
        self.state = ClientState::Disconnected;
        self.stream_key = None;
    }

    /// Returns and increments the internal chunk sequence number.
    fn next_sequence(&mut self) -> u32 {
        let n = self.chunk_sequence_number;
        self.chunk_sequence_number = self.chunk_sequence_number.wrapping_add(1);
        n
    }
}

/// AMF0 encoding helpers (no external dependencies).
pub mod amf0 {
    /// AMF0 type markers.
    const NUMBER_TYPE: u8 = 0x00;
    const BOOL_TYPE: u8 = 0x01;
    const STRING_TYPE: u8 = 0x02;
    const OBJECT_TYPE: u8 = 0x03;
    const NULL_TYPE: u8 = 0x05;
    const OBJECT_END_MARKER: [u8; 3] = [0x00, 0x00, 0x09];

    /// Encodes a UTF-8 string as AMF0 (type marker + 2-byte length + bytes).
    pub fn encode_string(s: &str) -> Vec<u8> {
        let bytes = s.as_bytes();
        let len = bytes.len() as u16;
        let mut buf = Vec::with_capacity(3 + bytes.len());
        buf.push(STRING_TYPE);
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(bytes);
        buf
    }

    /// Encodes a 64-bit IEEE-754 number as AMF0 (9 bytes total).
    pub fn encode_number(n: f64) -> Vec<u8> {
        let mut buf = Vec::with_capacity(9);
        buf.push(NUMBER_TYPE);
        buf.extend_from_slice(&n.to_bits().to_be_bytes());
        buf
    }

    /// Encodes a boolean as AMF0 (2 bytes total).
    pub fn encode_bool(b: bool) -> Vec<u8> {
        vec![BOOL_TYPE, if b { 1 } else { 0 }]
    }

    /// Encodes the AMF0 null value (1 byte).
    pub fn encode_null() -> Vec<u8> {
        vec![NULL_TYPE]
    }

    /// Returns the AMF0 object-start marker byte.
    pub fn start_object() -> Vec<u8> {
        vec![OBJECT_TYPE]
    }

    /// Returns the AMF0 object-end marker (3 bytes).
    pub fn end_object() -> Vec<u8> {
        OBJECT_END_MARKER.to_vec()
    }

    /// Encodes a key-value property for an AMF0 object.
    ///
    /// Format: 2-byte key length + key bytes + value bytes.
    /// (No type marker on the key — that is the AMF0 object property encoding.)
    pub fn encode_property(key: &str, value: &[u8]) -> Vec<u8> {
        let key_bytes = key.as_bytes();
        let key_len = key_bytes.len() as u16;
        let mut buf = Vec::with_capacity(2 + key_bytes.len() + value.len());
        buf.extend_from_slice(&key_len.to_be_bytes());
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(value);
        buf
    }
}

#[cfg(test)]
mod client_spec_tests {
    use super::*;

    #[test]
    fn test_stream_key_new() {
        let sk = StreamKey::new("live", "stream1");
        assert_eq!(sk.app, "live");
        assert_eq!(sk.key, "stream1");
    }

    #[test]
    fn test_stream_key_from_url() {
        let sk =
            StreamKey::from_url("rtmp://localhost/live/stream1").expect("should succeed in test");
        assert_eq!(sk.app, "live");
        assert_eq!(sk.key, "stream1");
    }

    #[test]
    fn test_publish_mode_str() {
        assert_eq!(PublishMode::Live.as_str(), "live");
        assert_eq!(PublishMode::Record.as_str(), "record");
        assert_eq!(PublishMode::Append.as_str(), "append");
    }

    #[test]
    fn test_client_state_is_active() {
        assert!(!ClientState::Disconnected.is_active());
        assert!(!ClientState::Connecting.is_active());
        assert!(!ClientState::Handshaking.is_active());
        assert!(ClientState::Connected.is_active());
        assert!(ClientState::Publishing.is_active());
        assert!(ClientState::Playing.is_active());
        assert!(!ClientState::Disconnecting.is_active());
        assert!(!ClientState::Error.is_active());
    }

    #[test]
    fn test_client_simulate_lifecycle() {
        let mut client = SimpleRtmpClient::with_default_config();
        assert_eq!(client.state(), ClientState::Disconnected);

        client.simulate_connect(StreamKey::new("live", "test"));
        assert_eq!(client.state(), ClientState::Connected);
        assert!(client.stream_key().is_some());

        client.simulate_publish();
        assert_eq!(client.state(), ClientState::Publishing);

        client.simulate_disconnect();
        assert_eq!(client.state(), ClientState::Disconnected);
        assert!(client.stream_key().is_none());
    }

    #[test]
    fn test_build_connect_message_not_empty() {
        let client = SimpleRtmpClient::with_default_config();
        let msg = client.build_connect_message("live", "rtmp://localhost/live");
        assert!(!msg.is_empty());
        // Should start with AMF0 string marker (0x02) for "connect"
        assert_eq!(msg[0], 0x02);
    }

    #[test]
    fn test_build_video_message() {
        let client = SimpleRtmpClient::with_default_config();
        let packet = RtmpMediaPacket {
            packet_type: RtmpMediaPacketType::VideoKeyframe,
            timestamp_ms: 1234,
            data: vec![0xAA, 0xBB, 0xCC],
            is_keyframe: true,
            composition_time_offset: 0,
        };
        let msg = client.build_video_message(&packet, 1);
        // First 4 bytes are the timestamp in big-endian
        assert_eq!(&msg[0..4], &1234u32.to_be_bytes());
        assert!(!msg.is_empty());
    }

    #[test]
    fn test_amf0_string() {
        let encoded = amf0::encode_string("hello");
        // Byte 0: type marker 0x02
        assert_eq!(encoded[0], 0x02);
        // Bytes 1-2: big-endian length = 5
        assert_eq!(encoded[1], 0x00);
        assert_eq!(encoded[2], 0x05);
        // Bytes 3-7: "hello"
        assert_eq!(&encoded[3..], b"hello");
    }

    #[test]
    fn test_amf0_number() {
        let encoded = amf0::encode_number(1.0);
        // AMF0 number is always 9 bytes (1 type byte + 8 data bytes)
        assert_eq!(encoded.len(), 9);
        assert_eq!(encoded[0], 0x00);
    }

    #[test]
    fn test_record_stats() {
        let mut client = SimpleRtmpClient::with_default_config();
        assert_eq!(client.stats().bytes_sent, 0);
        client.record_send(1024);
        assert_eq!(client.stats().bytes_sent, 1024);
        assert_eq!(client.stats().packets_sent, 1);
        client.record_send(512);
        assert_eq!(client.stats().bytes_sent, 1536);
        assert_eq!(client.stats().packets_sent, 2);
    }
}

/// Regression coverage for a real handshake-ordering bug in
/// [`RtmpClient::perform_handshake`]: the driver used to call `parse_s2()`
/// (which transitions the state machine to `Done`) *before* `generate_c2()`
/// (which transitions it to `AckSent`). Because `generate_c2` ran last, it
/// clobbered the terminal `Done` state back down to `AckSent`, so
/// `is_done()` permanently returned `false` after a fully successful
/// handshake and every real RTMP publish/ingest loopback (e.g. OBS/ffmpeg
/// against `oximedia-net`'s server) failed with "Handshake incomplete".
#[cfg(test)]
mod handshake_ordering_regression_tests {
    use super::*;
    use crate::rtmp::handshake::C0_SIZE;
    use tokio::net::TcpListener;

    /// Drives the *real* client handshake driver (`RtmpClient::perform_handshake`,
    /// private but reachable here since this module is a descendant of `client`)
    /// against a minimal server built directly on the `Handshake` primitives -
    /// mirroring `server::connection::ServerConnection::perform_handshake`
    /// byte-for-byte - over a genuine 127.0.0.1 TCP loopback socket. Asserts
    /// both sides finish in the `Done` state and that the C2/S2 echo payloads
    /// are byte-exact (C2 echoes S1's random data, S2 echoes C1's).
    #[tokio::test]
    async fn test_client_server_handshake_loopback_both_sides_reach_done() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral loopback listener");
        let addr = listener
            .local_addr()
            .expect("bound listener must have a local address");

        // Server task: mirrors `ServerConnection::perform_handshake`'s byte
        // sequence exactly (receive C0+C1 -> send S0+S1+S2 -> receive C2),
        // using the same `Handshake` primitives, over the real socket.
        let server_task = tokio::spawn(async move {
            let (mut socket, _) = listener
                .accept()
                .await
                .expect("accept incoming loopback connection");
            let mut server_hs = Handshake::new();
            server_hs.set_epoch(0x1234_5678);

            let mut c0c1 = vec![0u8; C0_SIZE + HANDSHAKE_SIZE];
            socket
                .read_exact(&mut c0c1)
                .await
                .expect("read C0+C1 from client");
            server_hs.parse_c0c1(&c0c1).expect("parse C0+C1");

            let s0s1s2 = server_hs.generate_s0s1s2();
            socket
                .write_all(&s0s1s2)
                .await
                .expect("write S0+S1+S2 to client");

            let mut c2 = vec![0u8; HANDSHAKE_SIZE];
            socket
                .read_exact(&mut c2)
                .await
                .expect("read C2 from client");
            server_hs.parse_c2(&c2).expect("parse C2");

            (server_hs.is_done(), c0c1, s0s1s2, c2)
        });

        // Client side: exercise the real (now-fixed) driver directly, exactly
        // as `RtmpClient::connect` does internally.
        let mut client = RtmpClient::new();
        client.stream = Some(
            TcpStream::connect(addr)
                .await
                .expect("connect to loopback listener"),
        );
        client
            .perform_handshake()
            .await
            .expect("client handshake must complete successfully");

        assert!(
            client.handshake.is_done(),
            "client handshake state must be Done after a successful exchange"
        );

        let (server_done, c0c1, s0s1s2, c2) = server_task.await.expect("server task panicked");
        assert!(
            server_done,
            "server handshake state must also be Done after a successful exchange"
        );

        // C2 must echo S1's random payload (not fabricated or swapped).
        let s1_random = &s0s1s2[C0_SIZE + 8..C0_SIZE + HANDSHAKE_SIZE];
        let c2_random = &c2[8..HANDSHAKE_SIZE];
        assert_eq!(
            c2_random, s1_random,
            "C2 random payload must echo the server's S1 random payload"
        );

        // S2 must echo C1's random payload (not fabricated or swapped).
        let c1_random = &c0c1[C0_SIZE + 8..C0_SIZE + HANDSHAKE_SIZE];
        let s2_start = C0_SIZE + HANDSHAKE_SIZE;
        let s2_random = &s0s1s2[s2_start + 8..s2_start + HANDSHAKE_SIZE];
        assert_eq!(
            s2_random, c1_random,
            "S2 random payload must echo the client's C1 random payload"
        );
    }
}
