//! NDI sender implementation
//!
//! This module handles sending NDI streams to receivers, including frame encoding,
//! multi-connection support, bandwidth adaptation, and tally handling.
#![allow(dead_code)]

use crate::discovery::{find_available_port, get_local_ip, NdiSourceInfo, SourceAnnouncer};
use crate::protocol::{
    current_timestamp, NdiAudioFrame, NdiConnection, NdiMetadata, NdiVideoFrame,
};
use crate::tally::TallyState;
use crate::{AudioFormat, NdiError, PtzCommand, Result, VideoFormat};
use bytes::Bytes;
use parking_lot::RwLock as ParkingLotRwLock;
use std::collections::HashMap;

// Use parking_lot RwLock everywhere except for connection state
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Notify};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, trace, warn};
use uuid::Uuid;
use ParkingLotRwLock as RwLock;

/// Sender configuration
#[derive(Debug, Clone)]
pub struct SenderConfig {
    /// Source name
    pub name: String,

    /// Groups this source belongs to
    pub groups: Vec<String>,

    /// Listening port (0 for automatic)
    pub port: u16,

    /// Maximum number of simultaneous connections
    pub max_connections: usize,

    /// Enable audio support
    pub enable_audio: bool,

    /// Enable video support
    pub enable_video: bool,

    /// Enable metadata support
    pub enable_metadata: bool,

    /// Enable tally support
    pub enable_tally: bool,

    /// Enable PTZ support
    pub enable_ptz: bool,

    /// Enable bandwidth adaptation
    pub enable_bandwidth_adaptation: bool,

    /// Low bandwidth threshold (bytes per second)
    pub low_bandwidth_threshold: u64,

    /// Heartbeat interval
    pub heartbeat_interval: Duration,

    /// Connection timeout
    pub connection_timeout: Duration,
}

impl Default for SenderConfig {
    fn default() -> Self {
        Self {
            name: "OxiMedia NDI Source".to_string(),
            groups: vec!["public".to_string()],
            port: 0,
            max_connections: 10,
            enable_audio: true,
            enable_video: true,
            enable_metadata: true,
            enable_tally: true,
            enable_ptz: true,
            enable_bandwidth_adaptation: true,
            low_bandwidth_threshold: 10_000_000, // 10 MB/s
            heartbeat_interval: Duration::from_secs(5),
            connection_timeout: Duration::from_secs(10),
        }
    }
}

/// Connection state for a single receiver
struct ConnectionState {
    /// Unique connection ID
    id: Uuid,

    /// Remote address
    address: SocketAddr,

    /// NDI connection
    connection: NdiConnection,

    /// Tally state from this receiver
    tally_state: TallyState,

    /// Last activity timestamp
    last_activity: std::time::Instant,

    /// Send task handle
    task_handle: Option<JoinHandle<()>>,

    /// Frame sender channel
    frame_tx: mpsc::UnboundedSender<SendFrame>,

    /// PTZ command sender
    ptz_tx: mpsc::UnboundedSender<PtzCommand>,
}

impl ConnectionState {
    /// Create a new connection state
    fn new(
        address: SocketAddr,
        connection: NdiConnection,
        frame_tx: mpsc::UnboundedSender<SendFrame>,
        ptz_tx: mpsc::UnboundedSender<PtzCommand>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            address,
            connection,
            tally_state: TallyState::default(),
            last_activity: std::time::Instant::now(),
            task_handle: None,
            frame_tx,
            ptz_tx,
        }
    }
}

/// Frame to send
#[derive(Debug, Clone)]
enum SendFrame {
    Video(NdiVideoFrame),
    Audio(NdiAudioFrame),
    Metadata(NdiMetadata, i64), // timestamp
}

/// Sender statistics
#[derive(Debug, Clone, Default)]
pub struct SenderStats {
    /// Total frames sent
    pub frames_sent: u64,
    /// Total video frames sent
    pub video_frames: u64,
    /// Total audio frames sent
    pub audio_frames: u64,
    /// Total metadata packets sent
    pub metadata_packets: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Current bitrate (bytes per second)
    pub bitrate: u64,
    /// Number of active connections
    pub active_connections: usize,
}

/// NDI sender implementation
pub struct NdiSender {
    /// Configuration
    config: SenderConfig,

    /// Source information
    source_info: Arc<NdiSourceInfo>,

    /// Source announcer
    announcer: Arc<RwLock<Option<SourceAnnouncer>>>,

    /// Active connections
    connections: Arc<RwLock<HashMap<Uuid, Arc<tokio::sync::RwLock<ConnectionState>>>>>,

    /// Statistics
    stats: Arc<RwLock<SenderStats>>,

    /// Combined tally state from all receivers
    combined_tally: Arc<RwLock<TallyState>>,

    /// PTZ command receiver
    ptz_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<PtzCommand>>>>,

    /// PTZ command sender (for external use)
    ptz_tx: mpsc::UnboundedSender<PtzCommand>,

    /// Shutdown notify
    shutdown: Arc<Notify>,

    /// Listener task handle
    listener_handle: Arc<RwLock<Option<JoinHandle<()>>>>,

    /// Video sequence number
    video_sequence: Arc<RwLock<u32>>,

    /// Audio sequence number
    audio_sequence: Arc<RwLock<u32>>,
}

impl NdiSender {
    /// Create a new NDI sender
    pub async fn new(config: SenderConfig) -> Result<Self> {
        // Get local IP and port
        let ip = get_local_ip()?;
        let port = if config.port == 0 {
            find_available_port()?
        } else {
            config.port
        };

        let address = SocketAddr::new(ip, port);

        // Create source info
        let source_info = NdiSourceInfo::new(config.name.clone(), address)
            .with_groups(config.groups.clone())
            .with_audio(config.enable_audio)
            .with_video(config.enable_video)
            .with_metadata(config.enable_metadata);

        // Create announcer
        let announcer = SourceAnnouncer::new(source_info.clone())?;
        announcer.announce()?;

        let (ptz_tx, ptz_rx) = mpsc::unbounded_channel();

        let sender = Self {
            config,
            source_info: Arc::new(source_info),
            announcer: Arc::new(RwLock::new(Some(announcer))),
            connections: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SenderStats::default())),
            combined_tally: Arc::new(RwLock::new(TallyState::default())),
            ptz_rx: Arc::new(RwLock::new(Some(ptz_rx))),
            ptz_tx,
            shutdown: Arc::new(Notify::new()),
            listener_handle: Arc::new(RwLock::new(None)),
            video_sequence: Arc::new(RwLock::new(0)),
            audio_sequence: Arc::new(RwLock::new(0)),
        };

        // Start listener
        sender.start_listener().await?;

        Ok(sender)
    }

    /// Start the connection listener
    async fn start_listener(&self) -> Result<()> {
        let address = self.source_info.address;
        let listener = TcpListener::bind(address)
            .await
            .map_err(|e| NdiError::Network(e))?;

        info!("NDI sender listening on {}", address);

        let connections = self.connections.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let ptz_tx = self.ptz_tx.clone();
        let shutdown = self.shutdown.clone();

        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = listener.accept() => {
                        match result {
                            Ok((stream, addr)) => {
                                info!("New NDI connection from {}", addr);

                                // Check max connections
                                if connections.read().len() >= config.max_connections {
                                    warn!("Max connections reached, rejecting {}", addr);
                                    continue;
                                }

                                // Handle connection
                                Self::handle_connection(
                                    stream,
                                    addr,
                                    connections.clone(),
                                    stats.clone(),
                                    ptz_tx.clone(),
                                );
                            }
                            Err(e) => {
                                error!("Failed to accept connection: {}", e);
                            }
                        }
                    }
                    _ = shutdown.notified() => {
                        info!("Listener shutting down");
                        break;
                    }
                }
            }
        });

        *self.listener_handle.write() = Some(handle);
        Ok(())
    }

    /// Handle a new connection
    fn handle_connection(
        stream: TcpStream,
        addr: SocketAddr,
        connections: Arc<RwLock<HashMap<Uuid, Arc<tokio::sync::RwLock<ConnectionState>>>>>,
        stats: Arc<RwLock<SenderStats>>,
        ptz_tx: mpsc::UnboundedSender<PtzCommand>,
    ) {
        let connection = NdiConnection::new(stream);
        let (frame_tx, mut frame_rx) = mpsc::unbounded_channel();
        let conn_id = Uuid::new_v4();

        // Create connection state with the generated UUID
        let conn_state = Arc::new(tokio::sync::RwLock::new(ConnectionState {
            id: conn_id,
            address: addr,
            connection,
            tally_state: TallyState::default(),
            last_activity: std::time::Instant::now(),
            task_handle: None,
            frame_tx,
            ptz_tx,
        }));

        connections.write().insert(conn_id, conn_state.clone());
        stats.write().active_connections = connections.read().len();

        // Spawn send task for this connection
        let connections_clone = connections.clone();
        let stats_clone = stats.clone();

        #[allow(clippy::type_complexity)]
        tokio::spawn(async move {
            loop {
                match frame_rx.recv().await {
                    Some(frame) => {
                        // Update timestamp
                        conn_state.write().await.last_activity = std::time::Instant::now();

                        let result = {
                            // Acquire lock briefly to send frame, but this still holds lock across await
                            // This is a limitation that would need deeper refactoring
                            let mut conn_write = conn_state.write().await;
                            let connection = &mut conn_write.connection;
                            match &frame {
                                SendFrame::Video(video) => {
                                    trace!("Sending video frame to {}", addr);
                                    connection
                                        .send_video_frame(
                                            video.format,
                                            video.data.clone(),
                                            video.stride,
                                            video.header.timestamp,
                                        )
                                        .await
                                }
                                SendFrame::Audio(audio) => {
                                    trace!("Sending audio frame to {}", addr);
                                    connection
                                        .send_audio_frame(
                                            audio.format,
                                            audio.data.clone(),
                                            audio.num_samples,
                                            audio.header.timestamp,
                                        )
                                        .await
                                }
                                SendFrame::Metadata(metadata, timestamp) => {
                                    trace!("Sending metadata to {}", addr);
                                    connection.send_metadata(metadata.clone(), *timestamp).await
                                }
                            }
                        };
                        if let Err(e) = result {
                            warn!("Failed to send to {}: {}", addr, e);
                            break;
                        }

                        // Update stats
                        let mut stats = stats_clone.write();
                        stats.frames_sent += 1;
                    }
                    None => {
                        debug!("Frame channel closed for {}", addr);
                        break;
                    }
                }
            }

            // Remove connection
            connections_clone.write().remove(&conn_id);
            stats_clone.write().active_connections = connections_clone.read().len();
            info!("Connection closed: {}", addr);
        });
    }

    /// Send a video frame to all connected receivers
    pub async fn send_video_frame(
        &self,
        format: VideoFormat,
        data: Bytes,
        stride: u32,
    ) -> Result<()> {
        if !self.config.enable_video {
            return Err(NdiError::Protocol("Video not enabled".to_string()));
        }

        let timestamp = current_timestamp();
        let sequence = {
            let mut seq = self.video_sequence.write();
            let current = *seq;
            *seq = seq.wrapping_add(1);
            current
        };

        let frame = NdiVideoFrame::new(sequence, timestamp, format, data, stride);
        self.broadcast_frame(SendFrame::Video(frame)).await
    }

    /// Send an audio frame to all connected receivers
    pub async fn send_audio_frame(
        &self,
        format: AudioFormat,
        data: Bytes,
        num_samples: u32,
    ) -> Result<()> {
        if !self.config.enable_audio {
            return Err(NdiError::Protocol("Audio not enabled".to_string()));
        }

        let timestamp = current_timestamp();
        let sequence = {
            let mut seq = self.audio_sequence.write();
            let current = *seq;
            *seq = seq.wrapping_add(1);
            current
        };

        let frame = NdiAudioFrame::new(sequence, timestamp, format, data, num_samples);
        self.broadcast_frame(SendFrame::Audio(frame)).await
    }

    /// Send metadata to all connected receivers
    pub async fn send_metadata(&self, metadata: NdiMetadata) -> Result<()> {
        if !self.config.enable_metadata {
            return Err(NdiError::Protocol("Metadata not enabled".to_string()));
        }

        let timestamp = current_timestamp();
        self.broadcast_frame(SendFrame::Metadata(metadata, timestamp))
            .await
    }

    /// Broadcast a frame to all connections
    async fn broadcast_frame(&self, frame: SendFrame) -> Result<()> {
        let connections = self.connections.read();

        if connections.is_empty() {
            return Ok(());
        }

        for conn in connections.values() {
            let conn = conn.read().await;
            if let Err(e) = conn.frame_tx.send(frame.clone()) {
                warn!("Failed to queue frame for {}: {}", conn.address, e);
            }
        }

        // Update stats
        let mut stats = self.stats.write();
        match &frame {
            SendFrame::Video(f) => {
                stats.video_frames += 1;
                stats.bytes_sent += f.data.len() as u64;
            }
            SendFrame::Audio(f) => {
                stats.audio_frames += 1;
                stats.bytes_sent += f.data.len() as u64;
            }
            SendFrame::Metadata(_, _) => {
                stats.metadata_packets += 1;
            }
        }

        Ok(())
    }

    /// Get sender statistics
    pub fn stats(&self) -> SenderStats {
        self.stats.read().clone()
    }

    /// Get the combined tally state from all receivers
    pub fn tally_state(&self) -> TallyState {
        *self.combined_tally.read()
    }

    /// Get the next PTZ command from any receiver
    pub async fn receive_ptz_command(&self) -> Option<PtzCommand> {
        if let Some(rx) = self.ptz_rx.write().as_mut() {
            rx.recv().await
        } else {
            None
        }
    }

    /// Try to receive a PTZ command without blocking
    pub fn try_receive_ptz_command(&self) -> Option<PtzCommand> {
        if let Some(rx) = self.ptz_rx.write().as_mut() {
            rx.try_recv().ok()
        } else {
            None
        }
    }

    /// Get the number of active connections
    pub fn connection_count(&self) -> usize {
        self.connections.read().len()
    }

    /// Get the list of connected receiver addresses
    pub async fn connected_addresses(&self) -> Vec<SocketAddr> {
        let connections = self.connections.read();
        let mut addresses = Vec::new();
        for conn in connections.values() {
            addresses.push(conn.read().await.address);
        }
        addresses
    }

    /// Get the source information
    pub fn source_info(&self) -> Arc<NdiSourceInfo> {
        self.source_info.clone()
    }

    /// Update the source configuration
    pub fn update_config(&self, config: SenderConfig) -> Result<()> {
        let new_source_info = NdiSourceInfo::new(config.name.clone(), self.source_info.address)
            .with_groups(config.groups.clone())
            .with_audio(config.enable_audio)
            .with_video(config.enable_video)
            .with_metadata(config.enable_metadata);

        if let Some(announcer) = self.announcer.write().as_mut() {
            announcer.update(new_source_info)?;
        }

        Ok(())
    }

    /// Disconnect all receivers and stop sending
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down NDI sender");

        // Signal shutdown
        self.shutdown.notify_waiters();

        // Close all connections
        self.connections.write().clear();

        // Stop listener
        if let Some(handle) = self.listener_handle.write().take() {
            let _ = handle.await;
        }

        // Unannounce
        if let Some(announcer) = self.announcer.write().take() {
            announcer.unannounce()?;
        }

        Ok(())
    }

    /// Get the listening address
    pub fn address(&self) -> SocketAddr {
        self.source_info.address
    }

    /// Check if any receivers are connected
    pub fn has_connections(&self) -> bool {
        !self.connections.read().is_empty()
    }

    /// Clean up stale connections
    pub fn cleanup_stale_connections(&self, timeout: Duration) {
        let mut connections = self.connections.write();
        let now = std::time::Instant::now();

        // Collect stale connection IDs using try_read (non-blocking)
        let stale_ids: Vec<_> = connections
            .iter()
            .filter_map(|(id, conn)| {
                conn.try_read().ok().and_then(|guard| {
                    let elapsed = now.duration_since(guard.last_activity);
                    if elapsed >= timeout {
                        Some(*id)
                    } else {
                        None
                    }
                })
            })
            .collect();

        for id in stale_ids {
            connections.remove(&id);
        }

        self.stats.write().active_connections = connections.len();
    }
}

impl Drop for NdiSender {
    fn drop(&mut self) {
        // Unannounce on drop
        if let Some(announcer) = self.announcer.write().take() {
            let _ = announcer.unannounce();
        }
    }
}

/// Helper function to convert RGB to YUV422
pub fn rgb_to_yuv422(rgb: &[u8], width: u32, height: u32) -> Bytes {
    let mut yuv = Vec::with_capacity((width * height * 2) as usize);

    for y in 0..height {
        for x in 0..(width / 2) {
            let idx = ((y * width + x * 2) * 3) as usize;

            if idx + 5 >= rgb.len() {
                break;
            }

            let r0 = i32::from(rgb[idx]);
            let g0 = i32::from(rgb[idx + 1]);
            let b0 = i32::from(rgb[idx + 2]);

            let r1 = i32::from(rgb[idx + 3]);
            let g1 = i32::from(rgb[idx + 4]);
            let b1 = i32::from(rgb[idx + 5]);

            let y0 = ((66 * r0 + 129 * g0 + 25 * b0 + 128) >> 8) + 16;
            let y1 = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16;

            let u = ((-38 * r0 - 74 * g0 + 112 * b0 + 128) >> 8) + 128;
            let v = ((112 * r0 - 94 * g0 - 18 * b0 + 128) >> 8) + 128;

            yuv.push(y0.clamp(0, 255) as u8);
            yuv.push(u.clamp(0, 255) as u8);
            yuv.push(y1.clamp(0, 255) as u8);
            yuv.push(v.clamp(0, 255) as u8);
        }
    }

    Bytes::from(yuv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sender_config_default() {
        let config = SenderConfig::default();
        assert_eq!(config.max_connections, 10);
        assert!(config.enable_video);
        assert!(config.enable_audio);
        assert!(config.enable_tally);
    }

    #[test]
    fn test_sender_stats() {
        let stats = SenderStats::default();
        assert_eq!(stats.frames_sent, 0);
        assert_eq!(stats.active_connections, 0);
    }

    #[test]
    fn test_rgb_to_yuv422() {
        let rgb = vec![
            255, 0, 0, // Red pixel 1
            255, 0, 0, // Red pixel 2
            0, 255, 0, // Green pixel 1
            0, 255, 0, // Green pixel 2
        ];
        let yuv = rgb_to_yuv422(&rgb, 4, 1);
        assert_eq!(yuv.len(), 8); // 4 pixels -> 8 bytes in YUV422
    }
}
