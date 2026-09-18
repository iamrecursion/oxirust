//! NDI receiver implementation
//!
//! This module handles receiving NDI streams from sources, including frame decoding,
//! audio/video synchronization, tally support, and PTZ control.
#![allow(dead_code)]

use crate::discovery::NdiSourceInfo;
use crate::protocol::{
    current_timestamp, FrameSynchronizer, NdiAudioFrame, NdiConnection, NdiFrame, NdiMetadata,
    NdiVideoFrame,
};
use crate::tally::TallyState;
use crate::{NdiError, PtzCommand, Result};
use bytes::Bytes;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::{mpsc, Notify};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, trace, warn};

/// Receiver configuration
#[derive(Debug, Clone)]
pub struct ReceiverConfig {
    /// Buffer size in frames
    pub buffer_size: usize,

    /// Maximum jitter buffer time in milliseconds
    pub jitter_buffer_ms: u64,

    /// Enable audio/video synchronization
    pub enable_sync: bool,

    /// Maximum sync offset in milliseconds
    pub max_sync_offset_ms: i64,

    /// Connection timeout
    pub connection_timeout: Duration,

    /// Receive timeout
    pub receive_timeout: Duration,

    /// Enable automatic reconnection
    pub auto_reconnect: bool,

    /// Reconnect delay
    pub reconnect_delay: Duration,

    /// Enable bandwidth adaptation
    pub enable_bandwidth_adaptation: bool,

    /// Low bandwidth mode threshold (bytes per second)
    pub low_bandwidth_threshold: u64,
}

impl Default for ReceiverConfig {
    fn default() -> Self {
        Self {
            buffer_size: 16,
            jitter_buffer_ms: 100,
            enable_sync: true,
            max_sync_offset_ms: 100,
            connection_timeout: Duration::from_secs(10),
            receive_timeout: Duration::from_secs(5),
            auto_reconnect: true,
            reconnect_delay: Duration::from_secs(2),
            enable_bandwidth_adaptation: true,
            low_bandwidth_threshold: 10_000_000, // 10 MB/s
        }
    }
}

/// NDI receiver state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiverState {
    /// Not connected
    Disconnected,
    /// Connecting to source
    Connecting,
    /// Connected and receiving
    Connected,
    /// Connection error
    Error,
}

/// Received frame data
#[derive(Debug, Clone)]
pub enum ReceivedFrame {
    /// Video frame
    Video(NdiVideoFrame),
    /// Audio frame
    Audio(NdiAudioFrame),
    /// Metadata
    Metadata(NdiMetadata),
}

/// Statistics for the receiver
#[derive(Debug, Clone, Default)]
pub struct ReceiverStats {
    /// Total frames received
    pub frames_received: u64,
    /// Total video frames received
    pub video_frames: u64,
    /// Total audio frames received
    pub audio_frames: u64,
    /// Total metadata packets received
    pub metadata_packets: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Current bitrate (bytes per second)
    pub bitrate: u64,
    /// Dropped frames
    pub dropped_frames: u64,
    /// Current buffer level
    pub buffer_level: usize,
    /// Average latency in microseconds
    pub avg_latency_us: i64,
}

/// NDI receiver implementation
pub struct NdiReceiver {
    /// Source information
    source_info: Arc<NdiSourceInfo>,

    /// Configuration
    config: ReceiverConfig,

    /// Current state
    state: Arc<RwLock<ReceiverState>>,

    /// Video frame queue
    video_queue: Arc<RwLock<VecDeque<NdiVideoFrame>>>,

    /// Audio frame queue
    audio_queue: Arc<RwLock<VecDeque<NdiAudioFrame>>>,

    /// Metadata queue
    metadata_queue: Arc<RwLock<VecDeque<NdiMetadata>>>,

    /// Frame synchronizer
    synchronizer: Arc<RwLock<Option<FrameSynchronizer>>>,

    /// Statistics
    stats: Arc<RwLock<ReceiverStats>>,

    /// Tally state
    tally_state: Arc<RwLock<TallyState>>,

    /// PTZ command sender
    ptz_tx: mpsc::UnboundedSender<PtzCommand>,

    /// PTZ command receiver (held by receiver task)
    ptz_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<PtzCommand>>>>,

    /// Shutdown notify
    shutdown: Arc<Notify>,

    /// Receiver task handle
    task_handle: Arc<RwLock<Option<JoinHandle<()>>>>,
}

impl NdiReceiver {
    /// Create a new NDI receiver
    pub async fn new(source_info: Arc<NdiSourceInfo>, config: ReceiverConfig) -> Result<Self> {
        let (ptz_tx, ptz_rx) = mpsc::unbounded_channel();

        let receiver = Self {
            source_info,
            config: config.clone(),
            state: Arc::new(RwLock::new(ReceiverState::Disconnected)),
            video_queue: Arc::new(RwLock::new(VecDeque::with_capacity(config.buffer_size))),
            audio_queue: Arc::new(RwLock::new(VecDeque::with_capacity(config.buffer_size))),
            metadata_queue: Arc::new(RwLock::new(VecDeque::with_capacity(16))),
            synchronizer: Arc::new(RwLock::new(if config.enable_sync {
                Some(FrameSynchronizer::new(config.max_sync_offset_ms))
            } else {
                None
            })),
            stats: Arc::new(RwLock::new(ReceiverStats::default())),
            tally_state: Arc::new(RwLock::new(TallyState::default())),
            ptz_tx,
            ptz_rx: Arc::new(RwLock::new(Some(ptz_rx))),
            shutdown: Arc::new(Notify::new()),
            task_handle: Arc::new(RwLock::new(None)),
        };

        // Start the receiver task
        receiver.start_receiver_task().await?;

        Ok(receiver)
    }

    /// Start the receiver task
    async fn start_receiver_task(&self) -> Result<()> {
        let source_info = self.source_info.clone();
        let config = self.config.clone();
        let state = self.state.clone();
        let video_queue = self.video_queue.clone();
        let audio_queue = self.audio_queue.clone();
        let metadata_queue = self.metadata_queue.clone();
        let synchronizer = self.synchronizer.clone();
        let stats = self.stats.clone();
        let tally_state = self.tally_state.clone();
        let ptz_rx = self.ptz_rx.write().take();
        let shutdown = self.shutdown.clone();

        let handle = tokio::spawn(async move {
            if let Some(mut ptz_rx) = ptz_rx {
                Self::receiver_task(
                    source_info,
                    config,
                    state,
                    video_queue,
                    audio_queue,
                    metadata_queue,
                    synchronizer,
                    stats,
                    tally_state,
                    &mut ptz_rx,
                    shutdown,
                )
                .await;
            }
        });

        *self.task_handle.write() = Some(handle);
        Ok(())
    }

    /// Receiver task implementation
    #[allow(clippy::too_many_arguments)]
    async fn receiver_task(
        source_info: Arc<NdiSourceInfo>,
        config: ReceiverConfig,
        state: Arc<RwLock<ReceiverState>>,
        video_queue: Arc<RwLock<VecDeque<NdiVideoFrame>>>,
        audio_queue: Arc<RwLock<VecDeque<NdiAudioFrame>>>,
        metadata_queue: Arc<RwLock<VecDeque<NdiMetadata>>>,
        synchronizer: Arc<RwLock<Option<FrameSynchronizer>>>,
        stats: Arc<RwLock<ReceiverStats>>,
        _tally_state: Arc<RwLock<TallyState>>,
        _ptz_rx: &mut mpsc::UnboundedReceiver<PtzCommand>,
        shutdown: Arc<Notify>,
    ) {
        loop {
            // Connect to source
            *state.write() = ReceiverState::Connecting;
            debug!("Connecting to NDI source: {}", source_info.name);

            match tokio::time::timeout(
                config.connection_timeout,
                TcpStream::connect(source_info.address),
            )
            .await
            {
                Ok(Ok(stream)) => {
                    info!("Connected to NDI source: {}", source_info.name);
                    *state.write() = ReceiverState::Connected;

                    let mut connection = NdiConnection::new(stream);

                    // Receive loop
                    if let Err(e) = Self::receive_loop(
                        &mut connection,
                        &config,
                        &video_queue,
                        &audio_queue,
                        &metadata_queue,
                        &synchronizer,
                        &stats,
                        &shutdown,
                    )
                    .await
                    {
                        warn!("Receive loop error: {}", e);
                        *state.write() = ReceiverState::Error;
                    }
                }
                Ok(Err(e)) => {
                    error!("Failed to connect to NDI source: {}", e);
                    *state.write() = ReceiverState::Error;
                }
                Err(_) => {
                    error!("Connection timeout");
                    *state.write() = ReceiverState::Error;
                }
            }

            // Check if we should reconnect
            if !config.auto_reconnect {
                break;
            }

            debug!("Reconnecting in {:?}", config.reconnect_delay);
            tokio::time::sleep(config.reconnect_delay).await;
        }

        *state.write() = ReceiverState::Disconnected;
    }

    /// Receive loop
    #[allow(clippy::too_many_arguments)]
    #[allow(unreachable_code)]
    async fn receive_loop(
        connection: &mut NdiConnection,
        config: &ReceiverConfig,
        video_queue: &Arc<RwLock<VecDeque<NdiVideoFrame>>>,
        audio_queue: &Arc<RwLock<VecDeque<NdiAudioFrame>>>,
        metadata_queue: &Arc<RwLock<VecDeque<NdiMetadata>>>,
        synchronizer: &Arc<RwLock<Option<FrameSynchronizer>>>,
        stats: &Arc<RwLock<ReceiverStats>>,
        _shutdown: &Arc<Notify>,
    ) -> Result<()> {
        let mut last_stats_update = std::time::Instant::now();
        let mut bytes_since_last_update = 0u64;

        loop {
            // Receive frame with timeout
            let frame = match connection
                .receive_frame_timeout(config.receive_timeout)
                .await
            {
                Ok(frame) => frame,
                Err(NdiError::Timeout) => {
                    // Send heartbeat on timeout
                    connection.send_heartbeat().await?;
                    continue;
                }
                Err(e) => return Err(e),
            };

            // Update stats
            let frame_size = match &frame {
                NdiFrame::Video(f) => f.data.len(),
                NdiFrame::Audio(f) => f.data.len(),
                NdiFrame::Metadata(_, _) => 0,
                NdiFrame::Heartbeat(_) => 0,
            };

            bytes_since_last_update += frame_size as u64;

            {
                let mut stats = stats.write();
                stats.frames_received += 1;
                stats.bytes_received += frame_size as u64;

                // Calculate latency
                let now = current_timestamp();
                let latency = now - frame.timestamp();
                stats.avg_latency_us = (stats.avg_latency_us * 9 + latency) / 10;
            }

            // Update bitrate every second
            if last_stats_update.elapsed() >= Duration::from_secs(1) {
                let mut stats = stats.write();
                stats.bitrate = bytes_since_last_update;
                bytes_since_last_update = 0;
                last_stats_update = std::time::Instant::now();
            }

            // Process frame
            match frame {
                NdiFrame::Video(video_frame) => {
                    trace!("Received video frame: seq={}", video_frame.header.sequence);
                    stats.write().video_frames += 1;

                    if let Some(sync) = synchronizer.write().as_mut() {
                        sync.add_video(video_frame);

                        // Try to get synchronized frames
                        if let Some((video, audio_frames)) = sync.get_synchronized() {
                            if let Some(v) = video {
                                Self::enqueue_video(video_queue, v, config.buffer_size, stats)?;
                            }
                            for a in audio_frames {
                                Self::enqueue_audio(audio_queue, a, config.buffer_size, stats)?;
                            }
                        }

                        // Clear old frames
                        sync.clear_old_frames(current_timestamp());
                    } else {
                        Self::enqueue_video(video_queue, video_frame, config.buffer_size, stats)?;
                    }
                }
                NdiFrame::Audio(audio_frame) => {
                    trace!("Received audio frame: seq={}", audio_frame.header.sequence);
                    stats.write().audio_frames += 1;

                    if let Some(sync) = synchronizer.write().as_mut() {
                        sync.add_audio(audio_frame);
                    } else {
                        Self::enqueue_audio(audio_queue, audio_frame, config.buffer_size, stats)?;
                    }
                }
                NdiFrame::Metadata(metadata, header) => {
                    trace!("Received metadata: seq={}", header.sequence);
                    stats.write().metadata_packets += 1;

                    let mut queue = metadata_queue.write();
                    if queue.len() >= 16 {
                        queue.pop_front();
                        stats.write().dropped_frames += 1;
                    }
                    queue.push_back(metadata);
                }
                NdiFrame::Heartbeat(_) => {
                    trace!("Received heartbeat");
                    // Respond with heartbeat
                    connection.send_heartbeat().await?;
                }
            }
        }

        Ok(())
    }

    /// Enqueue a video frame
    fn enqueue_video(
        queue: &Arc<RwLock<VecDeque<NdiVideoFrame>>>,
        frame: NdiVideoFrame,
        max_size: usize,
        stats: &Arc<RwLock<ReceiverStats>>,
    ) -> Result<()> {
        let mut queue = queue.write();
        if queue.len() >= max_size {
            queue.pop_front();
            stats.write().dropped_frames += 1;
        }
        queue.push_back(frame);
        stats.write().buffer_level = queue.len();
        Ok(())
    }

    /// Enqueue an audio frame
    fn enqueue_audio(
        queue: &Arc<RwLock<VecDeque<NdiAudioFrame>>>,
        frame: NdiAudioFrame,
        max_size: usize,
        stats: &Arc<RwLock<ReceiverStats>>,
    ) -> Result<()> {
        let mut queue = queue.write();
        if queue.len() >= max_size {
            queue.pop_front();
            stats.write().dropped_frames += 1;
        }
        queue.push_back(frame);
        Ok(())
    }

    /// Receive a video frame
    pub async fn receive_video(&self) -> Result<NdiVideoFrame> {
        loop {
            {
                let mut queue = self.video_queue.write();
                if let Some(frame) = queue.pop_front() {
                    self.stats.write().buffer_level = queue.len();
                    return Ok(frame);
                }
            }

            // Wait a bit and try again
            tokio::time::sleep(Duration::from_millis(1)).await;

            // Check state
            if *self.state.read() == ReceiverState::Disconnected {
                return Err(NdiError::ConnectionClosed);
            }
        }
    }

    /// Receive an audio frame
    pub async fn receive_audio(&self) -> Result<NdiAudioFrame> {
        loop {
            {
                let mut queue = self.audio_queue.write();
                if let Some(frame) = queue.pop_front() {
                    return Ok(frame);
                }
            }

            // Wait a bit and try again
            tokio::time::sleep(Duration::from_millis(1)).await;

            // Check state
            if *self.state.read() == ReceiverState::Disconnected {
                return Err(NdiError::ConnectionClosed);
            }
        }
    }

    /// Receive metadata
    pub async fn receive_metadata(&self) -> Result<NdiMetadata> {
        loop {
            {
                let mut queue = self.metadata_queue.write();
                if let Some(metadata) = queue.pop_front() {
                    return Ok(metadata);
                }
            }

            // Wait a bit and try again
            tokio::time::sleep(Duration::from_millis(10)).await;

            // Check state
            if *self.state.read() == ReceiverState::Disconnected {
                return Err(NdiError::ConnectionClosed);
            }
        }
    }

    /// Try to receive a video frame without blocking
    pub fn try_receive_video(&self) -> Option<NdiVideoFrame> {
        let mut queue = self.video_queue.write();
        let frame = queue.pop_front();
        if frame.is_some() {
            self.stats.write().buffer_level = queue.len();
        }
        frame
    }

    /// Try to receive an audio frame without blocking
    pub fn try_receive_audio(&self) -> Option<NdiAudioFrame> {
        self.audio_queue.write().pop_front()
    }

    /// Try to receive metadata without blocking
    pub fn try_receive_metadata(&self) -> Option<NdiMetadata> {
        self.metadata_queue.write().pop_front()
    }

    /// Get the current receiver state
    pub fn state(&self) -> ReceiverState {
        *self.state.read()
    }

    /// Get receiver statistics
    pub fn stats(&self) -> ReceiverStats {
        self.stats.read().clone()
    }

    /// Get the current tally state
    pub fn tally_state(&self) -> TallyState {
        *self.tally_state.read()
    }

    /// Set the tally state
    pub fn set_tally_state(&self, state: TallyState) {
        *self.tally_state.write() = state;
    }

    /// Send a PTZ command
    pub fn send_ptz_command(&self, command: PtzCommand) -> Result<()> {
        self.ptz_tx
            .send(command)
            .map_err(|_| NdiError::Protocol("Failed to send PTZ command".to_string()))
    }

    /// Get the number of video frames in the buffer
    pub fn video_buffer_len(&self) -> usize {
        self.video_queue.read().len()
    }

    /// Get the number of audio frames in the buffer
    pub fn audio_buffer_len(&self) -> usize {
        self.audio_queue.read().len()
    }

    /// Get the number of metadata packets in the buffer
    pub fn metadata_buffer_len(&self) -> usize {
        self.metadata_queue.read().len()
    }

    /// Clear all buffers
    pub fn clear_buffers(&self) {
        self.video_queue.write().clear();
        self.audio_queue.write().clear();
        self.metadata_queue.write().clear();
    }

    /// Get the source information
    pub fn source_info(&self) -> Arc<NdiSourceInfo> {
        self.source_info.clone()
    }

    /// Disconnect from the source
    pub async fn disconnect(&self) -> Result<()> {
        info!("Disconnecting from NDI source: {}", self.source_info.name);

        // Signal shutdown
        self.shutdown.notify_waiters();

        // Wait for task to finish
        if let Some(handle) = self.task_handle.write().take() {
            let _ = handle.await;
        }

        *self.state.write() = ReceiverState::Disconnected;
        Ok(())
    }

    /// Check if the receiver is connected
    pub fn is_connected(&self) -> bool {
        *self.state.read() == ReceiverState::Connected
    }

    /// Wait for connection
    pub async fn wait_for_connection(&self, timeout: Duration) -> Result<()> {
        let start = std::time::Instant::now();

        while start.elapsed() < timeout {
            if self.is_connected() {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Err(NdiError::Timeout)
    }

    /// Get the current latency in microseconds
    pub fn latency_us(&self) -> i64 {
        self.stats.read().avg_latency_us
    }

    /// Get the current bitrate in bytes per second
    pub fn bitrate(&self) -> u64 {
        self.stats.read().bitrate
    }
}

impl std::fmt::Debug for NdiReceiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NdiReceiver")
            .field("source_info", &self.source_info)
            .field("config", &self.config)
            .field("state", &self.state)
            .finish()
    }
}

/// Helper function to convert video frame data to RGB
pub fn yuv422_to_rgb(data: &[u8], width: u32, height: u32) -> Bytes {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);

    for y in 0..height {
        for x in 0..(width / 2) {
            let idx = ((y * width + x * 2) * 2) as usize;

            if idx + 3 >= data.len() {
                break;
            }

            let y0 = i32::from(data[idx]);
            let u = i32::from(data[idx + 1]);
            let y1 = i32::from(data[idx + 2]);
            let v = i32::from(data[idx + 3]);

            let c = y0 - 16;
            let d = u - 128;
            let e = v - 128;

            let r0 = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8;
            let g0 = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
            let b0 = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8;

            let c = y1 - 16;

            let r1 = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8;
            let g1 = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
            let b1 = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8;

            rgb.extend_from_slice(&[r0, g0, b0, r1, g1, b1]);
        }
    }

    Bytes::from(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receiver_config_default() {
        let config = ReceiverConfig::default();
        assert_eq!(config.buffer_size, 16);
        assert!(config.enable_sync);
        assert!(config.auto_reconnect);
    }

    #[test]
    fn test_receiver_stats() {
        let stats = ReceiverStats::default();
        assert_eq!(stats.frames_received, 0);
        assert_eq!(stats.video_frames, 0);
        assert_eq!(stats.audio_frames, 0);
    }

    #[test]
    fn test_yuv422_to_rgb() {
        let yuv = vec![
            128, 128, 128, 128, // Y0, U, Y1, V
            128, 128, 128, 128,
        ];
        let rgb = yuv422_to_rgb(&yuv, 4, 1);
        assert_eq!(rgb.len(), 12); // 4 pixels * 3 bytes
    }
}
