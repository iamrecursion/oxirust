//! Broadcast device control
//!
//! Support for professional broadcast device control protocols including:
//! - VDCP (Video Disk Control Protocol)
//! - Sony 9-pin (RS-422)
//! - ODetics protocol
//! - GPI/GPO (General Purpose Input/Output)
//! - Tally control
//! - Router control

use crate::{PlayoutError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info};

/// Device control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Enable device control
    pub enabled: bool,

    /// Device control protocol
    pub protocol: DeviceProtocol,

    /// Connection settings
    pub connection: ConnectionSettings,

    /// Command timeout in milliseconds
    pub command_timeout_ms: u64,

    /// Retry attempts on failure
    pub retry_attempts: u32,

    /// Heartbeat interval in seconds
    pub heartbeat_interval_sec: u32,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            protocol: DeviceProtocol::Vdcp,
            connection: ConnectionSettings::default(),
            command_timeout_ms: 5000,
            retry_attempts: 3,
            heartbeat_interval_sec: 10,
        }
    }
}

/// Connection settings for device control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionSettings {
    /// TCP/IP connection
    TcpIp {
        /// Remote address
        addr: SocketAddr,
        /// Keep-alive
        keepalive: bool,
    },
    /// Serial port connection
    Serial {
        /// Port name (e.g., "/dev/ttyUSB0")
        port: String,
        /// Baud rate
        baud_rate: u32,
        /// Data bits
        data_bits: u8,
        /// Parity
        parity: SerialParity,
        /// Stop bits
        stop_bits: u8,
    },
    /// UDP broadcast
    Udp {
        /// Bind address
        bind_addr: SocketAddr,
        /// Broadcast address
        broadcast_addr: SocketAddr,
    },
}

impl Default for ConnectionSettings {
    fn default() -> Self {
        Self::TcpIp {
            addr: SocketAddr::new(IpAddr::from([127, 0, 0, 1]), 4000),
            keepalive: true,
        }
    }
}

/// Serial parity settings
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SerialParity {
    None,
    Even,
    Odd,
    Mark,
    Space,
}

/// Device control protocols
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeviceProtocol {
    /// Video Disk Control Protocol
    Vdcp,
    /// Sony 9-pin (RS-422)
    Sony9Pin,
    /// ODetics protocol
    Odetics,
    /// Custom TCP/IP protocol
    Custom,
}

/// Device command types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceCommand {
    /// Play command
    Play {
        /// Clip ID
        clip_id: String,
        /// In point in frames
        in_point: Option<u64>,
        /// Out point in frames
        out_point: Option<u64>,
    },
    /// Stop command
    Stop,
    /// Cue command (prepare for play)
    Cue {
        /// Clip ID
        clip_id: String,
        /// Cue point in frames
        cue_point: u64,
    },
    /// Record command
    Record {
        /// Clip ID
        clip_id: String,
        /// Duration in frames
        duration: Option<u64>,
    },
    /// Fast forward
    FastForward {
        /// Speed multiplier
        speed: f32,
    },
    /// Rewind
    Rewind {
        /// Speed multiplier
        speed: f32,
    },
    /// Eject
    Eject,
    /// Status query
    StatusQuery,
    /// GPI trigger
    GpiTrigger {
        /// GPI port number
        port: u8,
        /// State (true = high, false = low)
        state: bool,
    },
    /// GPO trigger
    GpoTrigger {
        /// GPO port number
        port: u8,
        /// State (true = high, false = low)
        state: bool,
    },
    /// Router control
    RouterControl {
        /// Source input
        source: u16,
        /// Destination output
        destination: u16,
        /// Router level (video/audio)
        level: RouterLevel,
    },
    /// Tally control
    TallyControl {
        /// Camera/source ID
        source_id: u16,
        /// Tally state
        state: TallyState,
    },
}

/// Router control levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RouterLevel {
    Video,
    Audio,
    Both,
}

/// Tally states
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TallyState {
    Off,
    Preview,
    Program,
}

/// Device status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceStatus {
    /// Device is online
    pub online: bool,

    /// Current playback state
    pub playback_state: PlaybackState,

    /// Current timecode
    pub timecode: Option<String>,

    /// Active clip ID
    pub active_clip: Option<String>,

    /// Remaining time in frames
    pub remaining_frames: Option<u64>,

    /// Error state
    pub error: Option<String>,

    /// Last update timestamp
    pub last_update: chrono::DateTime<chrono::Utc>,
}

/// Playback state for devices
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlaybackState {
    Stopped,
    Playing,
    Paused,
    Cueing,
    FastForward,
    Rewind,
    Recording,
    Error,
}

/// VDCP protocol implementation
pub struct VdcpProtocol {
    config: DeviceConfig,
    connection: Option<TcpStream>,
    status: Arc<RwLock<DeviceStatus>>,
}

impl VdcpProtocol {
    /// Create new VDCP protocol handler
    pub fn new(config: DeviceConfig) -> Self {
        let status = DeviceStatus {
            online: false,
            playback_state: PlaybackState::Stopped,
            timecode: None,
            active_clip: None,
            remaining_frames: None,
            error: None,
            last_update: chrono::Utc::now(),
        };

        Self {
            config,
            connection: None,
            status: Arc::new(RwLock::new(status)),
        }
    }

    /// Connect to VDCP device
    pub async fn connect(&mut self) -> Result<()> {
        if let ConnectionSettings::TcpIp { addr, .. } = &self.config.connection {
            let stream = TcpStream::connect(addr).await.map_err(|e| {
                PlayoutError::Output(format!("Failed to connect to VDCP device: {e}"))
            })?;

            self.connection = Some(stream);

            let mut status = self.status.write().await;
            status.online = true;
            status.last_update = chrono::Utc::now();

            info!("Connected to VDCP device at {}", addr);
            Ok(())
        } else {
            Err(PlayoutError::Config(
                "VDCP requires TCP/IP connection".to_string(),
            ))
        }
    }

    /// Send VDCP command
    pub async fn send_command(&mut self, command: &DeviceCommand) -> Result<()> {
        let packet = self.encode_vdcp_command(command)?;

        let stream = self
            .connection
            .as_mut()
            .ok_or_else(|| PlayoutError::Output("Not connected to VDCP device".to_string()))?;

        stream
            .write_all(&packet)
            .await
            .map_err(|e| PlayoutError::Output(format!("Failed to send VDCP command: {e}")))?;

        debug!("Sent VDCP command: {:?}", command);
        Ok(())
    }

    /// Encode VDCP command to binary packet
    fn encode_vdcp_command(&self, command: &DeviceCommand) -> Result<Vec<u8>> {
        let mut packet = Vec::new();

        // VDCP header
        packet.push(0x80); // Start byte

        match command {
            DeviceCommand::Play {
                clip_id,
                in_point,
                out_point,
            } => {
                packet.push(0x01); // Play command
                packet.extend_from_slice(clip_id.as_bytes());
                if let Some(ip) = in_point {
                    packet.extend_from_slice(&ip.to_be_bytes());
                }
                if let Some(op) = out_point {
                    packet.extend_from_slice(&op.to_be_bytes());
                }
            }
            DeviceCommand::Stop => {
                packet.push(0x02); // Stop command
            }
            DeviceCommand::Cue { clip_id, cue_point } => {
                packet.push(0x03); // Cue command
                packet.extend_from_slice(clip_id.as_bytes());
                packet.extend_from_slice(&cue_point.to_be_bytes());
            }
            DeviceCommand::Record { clip_id, duration } => {
                packet.push(0x04); // Record command
                packet.extend_from_slice(clip_id.as_bytes());
                if let Some(dur) = duration {
                    packet.extend_from_slice(&dur.to_be_bytes());
                }
            }
            DeviceCommand::StatusQuery => {
                packet.push(0x10); // Status query
            }
            _ => {
                return Err(PlayoutError::Output(
                    "Command not supported by VDCP protocol".to_string(),
                ));
            }
        }

        // Add checksum
        let checksum = packet.iter().fold(0u8, |acc, &x| acc.wrapping_add(x));
        packet.push(checksum);

        Ok(packet)
    }

    /// Read VDCP response
    pub async fn read_response(&mut self) -> Result<DeviceStatus> {
        let stream = self
            .connection
            .as_mut()
            .ok_or_else(|| PlayoutError::Output("Not connected to VDCP device".to_string()))?;

        let mut buffer = [0u8; 256];
        let n = stream
            .read(&mut buffer)
            .await
            .map_err(|e| PlayoutError::Output(format!("Failed to read VDCP response: {e}")))?;

        if n == 0 {
            return Err(PlayoutError::Output("Connection closed".to_string()));
        }

        self.decode_vdcp_response(&buffer[..n]).await
    }

    /// Decode VDCP response
    async fn decode_vdcp_response(&self, data: &[u8]) -> Result<DeviceStatus> {
        if data.is_empty() || data[0] != 0x80 {
            return Err(PlayoutError::Output("Invalid VDCP response".to_string()));
        }

        let mut status = self.status.write().await;
        status.last_update = chrono::Utc::now();

        if data.len() > 1 {
            status.playback_state = match data[1] {
                0x01 => PlaybackState::Playing,
                0x02 => PlaybackState::Stopped,
                0x03 => PlaybackState::Paused,
                0x04 => PlaybackState::Recording,
                _ => PlaybackState::Error,
            };
        }

        Ok(status.clone())
    }

    /// Get current device status
    pub async fn status(&self) -> DeviceStatus {
        self.status.read().await.clone()
    }
}

/// Sony 9-pin protocol implementation
pub struct Sony9PinProtocol {
    config: DeviceConfig,
    connection: Option<TcpStream>,
    status: Arc<RwLock<DeviceStatus>>,
}

impl Sony9PinProtocol {
    /// Create new Sony 9-pin protocol handler
    pub fn new(config: DeviceConfig) -> Self {
        let status = DeviceStatus {
            online: false,
            playback_state: PlaybackState::Stopped,
            timecode: None,
            active_clip: None,
            remaining_frames: None,
            error: None,
            last_update: chrono::Utc::now(),
        };

        Self {
            config,
            connection: None,
            status: Arc::new(RwLock::new(status)),
        }
    }

    /// Connect to Sony 9-pin device
    pub async fn connect(&mut self) -> Result<()> {
        if let ConnectionSettings::TcpIp { addr, .. } = &self.config.connection {
            let stream = TcpStream::connect(addr).await.map_err(|e| {
                PlayoutError::Output(format!("Failed to connect to Sony 9-pin device: {e}"))
            })?;

            self.connection = Some(stream);

            let mut status = self.status.write().await;
            status.online = true;
            status.last_update = chrono::Utc::now();

            info!("Connected to Sony 9-pin device at {}", addr);
            Ok(())
        } else {
            Err(PlayoutError::Config(
                "Sony 9-pin requires TCP/IP connection".to_string(),
            ))
        }
    }

    /// Send Sony 9-pin command
    pub async fn send_command(&mut self, command: &DeviceCommand) -> Result<()> {
        let packet = self.encode_sony_command(command)?;

        let stream = self.connection.as_mut().ok_or_else(|| {
            PlayoutError::Output("Not connected to Sony 9-pin device".to_string())
        })?;

        stream
            .write_all(&packet)
            .await
            .map_err(|e| PlayoutError::Output(format!("Failed to send Sony 9-pin command: {e}")))?;

        debug!("Sent Sony 9-pin command: {:?}", command);
        Ok(())
    }

    /// Encode Sony 9-pin command
    fn encode_sony_command(&self, command: &DeviceCommand) -> Result<Vec<u8>> {
        let mut packet = Vec::new();

        // Sony 9-pin packet format: CMD1 CMD2 DATA1 DATA2 DATA3 DATA4
        match command {
            DeviceCommand::Play { .. } => {
                packet.extend_from_slice(&[0x20, 0x01]); // Play command
            }
            DeviceCommand::Stop => {
                packet.extend_from_slice(&[0x20, 0x00]); // Stop command
            }
            DeviceCommand::Record { .. } => {
                packet.extend_from_slice(&[0x20, 0x02]); // Record command
            }
            DeviceCommand::FastForward { speed } => {
                packet.extend_from_slice(&[0x21, 0x10]); // FF command
                packet.push((speed * 10.0) as u8);
            }
            DeviceCommand::Rewind { speed } => {
                packet.extend_from_slice(&[0x21, 0x20]); // Rewind command
                packet.push((speed * 10.0) as u8);
            }
            DeviceCommand::Eject => {
                packet.extend_from_slice(&[0x20, 0x0F]); // Eject command
            }
            DeviceCommand::StatusQuery => {
                packet.extend_from_slice(&[0x61, 0x0A]); // Status sense
            }
            _ => {
                return Err(PlayoutError::Output(
                    "Command not supported by Sony 9-pin protocol".to_string(),
                ));
            }
        }

        // Calculate checksum
        let checksum = packet.iter().fold(0u8, |acc, &x| acc.wrapping_add(x));
        packet.push(checksum);

        Ok(packet)
    }

    /// Get current device status
    pub async fn status(&self) -> DeviceStatus {
        self.status.read().await.clone()
    }
}

/// GPI/GPO controller
pub struct GpioController {
    #[allow(dead_code)]
    config: DeviceConfig,
    inputs: Arc<RwLock<HashMap<u8, bool>>>,
    outputs: Arc<RwLock<HashMap<u8, bool>>>,
    event_tx: mpsc::Sender<GpioEvent>,
}

/// GPIO event
#[derive(Debug, Clone)]
pub struct GpioEvent {
    /// Port number
    pub port: u8,
    /// Event type
    pub event_type: GpioEventType,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// GPIO event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpioEventType {
    /// Input went high
    InputHigh,
    /// Input went low
    InputLow,
    /// Output set high
    OutputHigh,
    /// Output set low
    OutputLow,
}

impl GpioController {
    /// Create new GPIO controller
    pub fn new(config: DeviceConfig, event_tx: mpsc::Sender<GpioEvent>) -> Self {
        Self {
            config,
            inputs: Arc::new(RwLock::new(HashMap::new())),
            outputs: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
        }
    }

    /// Read GPI input state
    pub async fn read_gpi(&self, port: u8) -> Option<bool> {
        self.inputs.read().await.get(&port).copied()
    }

    /// Set GPO output state
    pub async fn set_gpo(&self, port: u8, state: bool) -> Result<()> {
        {
            let mut outputs = self.outputs.write().await;
            outputs.insert(port, state);
        }

        let event = GpioEvent {
            port,
            event_type: if state {
                GpioEventType::OutputHigh
            } else {
                GpioEventType::OutputLow
            },
            timestamp: chrono::Utc::now(),
        };

        self.event_tx
            .send(event)
            .await
            .map_err(|e| PlayoutError::Output(format!("Failed to send GPIO event: {e}")))?;

        debug!("Set GPO port {} to {}", port, state);
        Ok(())
    }

    /// Update GPI input state (called by monitoring task)
    pub async fn update_gpi(&self, port: u8, state: bool) -> Result<()> {
        let previous_state = self.inputs.read().await.get(&port).copied();

        if previous_state != Some(state) {
            {
                let mut inputs = self.inputs.write().await;
                inputs.insert(port, state);
            }

            let event = GpioEvent {
                port,
                event_type: if state {
                    GpioEventType::InputHigh
                } else {
                    GpioEventType::InputLow
                },
                timestamp: chrono::Utc::now(),
            };

            self.event_tx
                .send(event)
                .await
                .map_err(|e| PlayoutError::Output(format!("Failed to send GPIO event: {e}")))?;

            info!("GPI port {} changed to {}", port, state);
        }

        Ok(())
    }
}

/// Router controller for video/audio routing
pub struct RouterController {
    config: DeviceConfig,
    crosspoints: Arc<RwLock<HashMap<u16, u16>>>,
    connection: Option<TcpStream>,
}

impl RouterController {
    /// Create new router controller
    pub fn new(config: DeviceConfig) -> Self {
        Self {
            config,
            crosspoints: Arc::new(RwLock::new(HashMap::new())),
            connection: None,
        }
    }

    /// Connect to router
    pub async fn connect(&mut self) -> Result<()> {
        if let ConnectionSettings::TcpIp { addr, .. } = &self.config.connection {
            let stream = TcpStream::connect(addr)
                .await
                .map_err(|e| PlayoutError::Output(format!("Failed to connect to router: {e}")))?;

            self.connection = Some(stream);
            info!("Connected to router at {}", addr);
            Ok(())
        } else {
            Err(PlayoutError::Config(
                "Router requires TCP/IP connection".to_string(),
            ))
        }
    }

    /// Set router crosspoint (source to destination)
    pub async fn set_crosspoint(
        &mut self,
        source: u16,
        destination: u16,
        level: RouterLevel,
    ) -> Result<()> {
        let stream = self
            .connection
            .as_mut()
            .ok_or_else(|| PlayoutError::Output("Not connected to router".to_string()))?;

        // Simple router protocol: DEST SOURCE LEVEL
        let packet = match level {
            RouterLevel::Video => vec![
                0x01,
                (destination >> 8) as u8,
                destination as u8,
                (source >> 8) as u8,
                source as u8,
            ],
            RouterLevel::Audio => vec![
                0x02,
                (destination >> 8) as u8,
                destination as u8,
                (source >> 8) as u8,
                source as u8,
            ],
            RouterLevel::Both => vec![
                0x03,
                (destination >> 8) as u8,
                destination as u8,
                (source >> 8) as u8,
                source as u8,
            ],
        };

        stream
            .write_all(&packet)
            .await
            .map_err(|e| PlayoutError::Output(format!("Failed to set router crosspoint: {e}")))?;

        {
            let mut crosspoints = self.crosspoints.write().await;
            crosspoints.insert(destination, source);
        }

        info!(
            "Set router crosspoint: source {} -> destination {} (level: {:?})",
            source, destination, level
        );
        Ok(())
    }

    /// Get current crosspoint for destination
    pub async fn get_crosspoint(&self, destination: u16) -> Option<u16> {
        self.crosspoints.read().await.get(&destination).copied()
    }
}

/// Tally controller
pub struct TallyController {
    states: Arc<RwLock<HashMap<u16, TallyState>>>,
    connection: Option<TcpStream>,
}

impl TallyController {
    /// Create new tally controller
    pub fn new() -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
            connection: None,
        }
    }

    /// Set tally state for source
    pub async fn set_tally(&mut self, source_id: u16, state: TallyState) -> Result<()> {
        {
            let mut states = self.states.write().await;
            states.insert(source_id, state);
        }

        // Send tally command if connected
        if let Some(stream) = &mut self.connection {
            let packet = vec![
                0xFF, // Tally command
                (source_id >> 8) as u8,
                source_id as u8,
                match state {
                    TallyState::Off => 0x00,
                    TallyState::Preview => 0x01,
                    TallyState::Program => 0x02,
                },
            ];

            stream
                .write_all(&packet)
                .await
                .map_err(|e| PlayoutError::Output(format!("Failed to set tally: {e}")))?;
        }

        debug!("Set tally for source {} to {:?}", source_id, state);
        Ok(())
    }

    /// Get tally state for source
    pub async fn get_tally(&self, source_id: u16) -> Option<TallyState> {
        self.states.read().await.get(&source_id).copied()
    }
}

impl Default for TallyController {
    fn default() -> Self {
        Self::new()
    }
}

/// Device manager to coordinate all device controllers
pub struct DeviceManager {
    vdcp: Option<VdcpProtocol>,
    sony: Option<Sony9PinProtocol>,
    gpio: Option<GpioController>,
    router: Option<RouterController>,
    tally: Option<TallyController>,
}

impl DeviceManager {
    /// Create new device manager
    pub fn new() -> Self {
        Self {
            vdcp: None,
            sony: None,
            gpio: None,
            router: None,
            tally: None,
        }
    }

    /// Initialize VDCP controller
    pub fn init_vdcp(&mut self, config: DeviceConfig) {
        self.vdcp = Some(VdcpProtocol::new(config));
    }

    /// Initialize Sony 9-pin controller
    pub fn init_sony(&mut self, config: DeviceConfig) {
        self.sony = Some(Sony9PinProtocol::new(config));
    }

    /// Initialize GPIO controller
    pub fn init_gpio(&mut self, config: DeviceConfig, event_tx: mpsc::Sender<GpioEvent>) {
        self.gpio = Some(GpioController::new(config, event_tx));
    }

    /// Initialize router controller
    pub fn init_router(&mut self, config: DeviceConfig) {
        self.router = Some(RouterController::new(config));
    }

    /// Initialize tally controller
    pub fn init_tally(&mut self) {
        self.tally = Some(TallyController::new());
    }

    /// Connect all initialized devices
    pub async fn connect_all(&mut self) -> Result<()> {
        if let Some(vdcp) = &mut self.vdcp {
            vdcp.connect().await?;
        }

        if let Some(sony) = &mut self.sony {
            sony.connect().await?;
        }

        if let Some(router) = &mut self.router {
            router.connect().await?;
        }

        info!("All device controllers connected");
        Ok(())
    }

    /// Send command to appropriate device
    pub async fn send_command(
        &mut self,
        protocol: DeviceProtocol,
        command: DeviceCommand,
    ) -> Result<()> {
        match protocol {
            DeviceProtocol::Vdcp => {
                if let Some(vdcp) = &mut self.vdcp {
                    vdcp.send_command(&command).await?;
                } else {
                    return Err(PlayoutError::Config(
                        "VDCP controller not initialized".to_string(),
                    ));
                }
            }
            DeviceProtocol::Sony9Pin => {
                if let Some(sony) = &mut self.sony {
                    sony.send_command(&command).await?;
                } else {
                    return Err(PlayoutError::Config(
                        "Sony 9-pin controller not initialized".to_string(),
                    ));
                }
            }
            _ => {
                return Err(PlayoutError::Config(format!(
                    "Protocol {protocol:?} not supported"
                )));
            }
        }

        Ok(())
    }

    /// Get VDCP controller reference
    pub fn vdcp(&self) -> Option<&VdcpProtocol> {
        self.vdcp.as_ref()
    }

    /// Get Sony controller reference
    pub fn sony(&self) -> Option<&Sony9PinProtocol> {
        self.sony.as_ref()
    }

    /// Get GPIO controller reference
    pub fn gpio(&self) -> Option<&GpioController> {
        self.gpio.as_ref()
    }

    /// Get router controller reference
    pub fn router(&self) -> Option<&RouterController> {
        self.router.as_ref()
    }

    /// Get tally controller reference
    pub fn tally(&self) -> Option<&TallyController> {
        self.tally.as_ref()
    }

    /// Get mutable router controller reference
    pub fn router_mut(&mut self) -> Option<&mut RouterController> {
        self.router.as_mut()
    }

    /// Get mutable tally controller reference
    pub fn tally_mut(&mut self) -> Option<&mut TallyController> {
        self.tally.as_mut()
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_config_default() {
        let config = DeviceConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.protocol, DeviceProtocol::Vdcp);
        assert_eq!(config.command_timeout_ms, 5000);
    }

    #[test]
    fn test_vdcp_encode_play_command() {
        let config = DeviceConfig::default();
        let protocol = VdcpProtocol::new(config);

        let command = DeviceCommand::Play {
            clip_id: "TEST001".to_string(),
            in_point: Some(100),
            out_point: Some(200),
        };

        let result = protocol.encode_vdcp_command(&command);
        assert!(result.is_ok());
        let packet = result.expect("should succeed in test");
        assert_eq!(packet[0], 0x80); // Start byte
        assert_eq!(packet[1], 0x01); // Play command
    }

    #[test]
    fn test_vdcp_encode_stop_command() {
        let config = DeviceConfig::default();
        let protocol = VdcpProtocol::new(config);

        let command = DeviceCommand::Stop;
        let result = protocol.encode_vdcp_command(&command);
        assert!(result.is_ok());
        let packet = result.expect("should succeed in test");
        assert_eq!(packet[0], 0x80);
        assert_eq!(packet[1], 0x02); // Stop command
    }

    #[test]
    fn test_sony_encode_play_command() {
        let config = DeviceConfig::default();
        let protocol = Sony9PinProtocol::new(config);

        let command = DeviceCommand::Play {
            clip_id: "TEST001".to_string(),
            in_point: None,
            out_point: None,
        };

        let result = protocol.encode_sony_command(&command);
        assert!(result.is_ok());
        let packet = result.expect("should succeed in test");
        assert_eq!(packet[0], 0x20);
        assert_eq!(packet[1], 0x01);
    }

    #[tokio::test]
    async fn test_gpio_controller() {
        let (tx, mut rx) = mpsc::channel(10);
        let config = DeviceConfig::default();
        let controller = GpioController::new(config, tx);

        // Test GPO set
        controller
            .set_gpo(1, true)
            .await
            .expect("should succeed in test");

        // Verify event was sent
        let event = rx.recv().await.expect("should succeed in test");
        assert_eq!(event.port, 1);
        assert_eq!(event.event_type, GpioEventType::OutputHigh);
    }

    #[tokio::test]
    async fn test_gpio_gpi_update() {
        let (tx, mut rx) = mpsc::channel(10);
        let config = DeviceConfig::default();
        let controller = GpioController::new(config, tx);

        // Test GPI update
        controller
            .update_gpi(2, true)
            .await
            .expect("should succeed in test");

        // Verify event was sent
        let event = rx.recv().await.expect("should succeed in test");
        assert_eq!(event.port, 2);
        assert_eq!(event.event_type, GpioEventType::InputHigh);

        // Verify state was stored
        let state = controller.read_gpi(2).await;
        assert_eq!(state, Some(true));
    }

    #[tokio::test]
    async fn test_tally_controller() {
        let mut controller = TallyController::new();

        // Test setting tally state
        controller
            .set_tally(1, TallyState::Program)
            .await
            .expect("should succeed in test");

        // Verify state was stored
        let state = controller.get_tally(1).await;
        assert_eq!(state, Some(TallyState::Program));
    }

    #[tokio::test]
    async fn test_router_controller() {
        let config = DeviceConfig::default();
        let controller = RouterController::new(config);

        // Verify initial state
        let crosspoint = controller.get_crosspoint(1).await;
        assert_eq!(crosspoint, None);
    }

    #[test]
    fn test_device_manager_initialization() {
        let manager = DeviceManager::new();
        assert!(manager.vdcp().is_none());
        assert!(manager.sony().is_none());
        assert!(manager.gpio().is_none());
        assert!(manager.router().is_none());
        assert!(manager.tally().is_none());
    }

    #[test]
    fn test_device_manager_init_vdcp() {
        let mut manager = DeviceManager::new();
        let config = DeviceConfig::default();
        manager.init_vdcp(config);
        assert!(manager.vdcp().is_some());
    }

    #[test]
    fn test_playback_state_equality() {
        assert_eq!(PlaybackState::Playing, PlaybackState::Playing);
        assert_ne!(PlaybackState::Playing, PlaybackState::Stopped);
    }

    #[test]
    fn test_router_level_equality() {
        assert_eq!(RouterLevel::Video, RouterLevel::Video);
        assert_ne!(RouterLevel::Video, RouterLevel::Audio);
    }

    #[test]
    fn test_tally_state_values() {
        let states = vec![TallyState::Off, TallyState::Preview, TallyState::Program];
        assert_eq!(states.len(), 3);
    }
}
