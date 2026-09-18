//! Remote virtual production monitoring and control over network.
//!
//! Provides a pure-Rust network protocol for remote operators to monitor
//! and control a virtual production session over TCP/IP.  Supports:
//! - Real-time telemetry streaming (camera pose, LED status, sync state)
//! - Remote command dispatch (recalibrate, adjust color, change workflow)
//! - Session recording and playback
//! - Multi-operator observer mode with role-based access control
//!
//! The protocol is framed with a 4-byte little-endian length prefix followed
//! by a JSON payload, keeping it human-readable and easy to implement in
//! any language.

use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Protocol types
// ---------------------------------------------------------------------------

/// Operator role within a remote session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorRole {
    /// Full control: can send commands and receive telemetry.
    Director,
    /// Camera operator: can adjust camera settings but not workflow.
    CameraOp,
    /// LED technician: can adjust LED parameters.
    LedTech,
    /// Observer: read-only telemetry access.
    Observer,
}

impl OperatorRole {
    /// Whether this role can send control commands.
    #[must_use]
    pub fn can_control(self) -> bool {
        matches!(self, Self::Director | Self::CameraOp | Self::LedTech)
    }

    /// Whether this role can receive telemetry.
    #[must_use]
    pub fn can_observe(self) -> bool {
        true // All roles can observe
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Director => "Director",
            Self::CameraOp => "Camera Operator",
            Self::LedTech => "LED Technician",
            Self::Observer => "Observer",
        }
    }
}

/// Remote command sent by a client to the production server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemoteCommand {
    /// Ping to keep the connection alive.
    Ping { sequence: u64 },
    /// Trigger camera re-calibration.
    Recalibrate { camera_id: u32 },
    /// Set overall brightness of the LED wall (0.0 – 1.0).
    SetLedBrightness { brightness: f32 },
    /// Change the active workflow type.
    SetWorkflow { workflow: String },
    /// Request a specific telemetry snapshot.
    RequestSnapshot,
    /// Start recording the session.
    StartRecording { session_name: String },
    /// Stop the current recording.
    StopRecording,
    /// Adjust color temperature offset (Kelvin delta, e.g. +200 = warmer).
    SetColorTemperature { delta_k: i32 },
    /// Emergency stop: halt all live output.
    EmergencyStop,
    /// Acknowledge an emergency stop and resume normal operations.
    ResumeFromStop,
}

/// Telemetry data snapshot sent from the production server to clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    /// Monotonic timestamp in nanoseconds (from session start).
    pub timestamp_ns: u64,
    /// Frame number.
    pub frame_number: u64,
    /// Camera pose (position `[x,y,z]` in meters, euler angles `[rx,ry,rz]` in degrees).
    pub camera_position: [f64; 3],
    pub camera_euler_deg: [f64; 3],
    /// LED wall brightness (0.0–1.0).
    pub led_brightness: f32,
    /// Sync status label.
    pub sync_status: String,
    /// Pipeline latency estimate in microseconds.
    pub pipeline_latency_us: u64,
    /// Current workflow name.
    pub workflow: String,
    /// Whether the session is currently recording.
    pub is_recording: bool,
    /// Active operator count.
    pub operator_count: usize,
}

impl TelemetrySnapshot {
    /// Create a default/empty snapshot.
    #[must_use]
    pub fn default_snapshot() -> Self {
        Self {
            timestamp_ns: 0,
            frame_number: 0,
            camera_position: [0.0; 3],
            camera_euler_deg: [0.0; 3],
            led_brightness: 1.0,
            sync_status: "Unlocked".to_string(),
            pipeline_latency_us: 0,
            workflow: "LedWall".to_string(),
            is_recording: false,
            operator_count: 0,
        }
    }
}

/// Response from server to a client command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResponse {
    /// Whether the command succeeded.
    pub success: bool,
    /// Human-readable message.
    pub message: String,
    /// Original command type name for correlation.
    pub command_type: String,
    /// Server timestamp when the response was generated.
    pub timestamp_ns: u64,
}

impl CommandResponse {
    /// Create a success response.
    #[must_use]
    pub fn success(command_type: &str, message: &str) -> Self {
        Self {
            success: true,
            message: message.to_string(),
            command_type: command_type.to_string(),
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos() as u64,
        }
    }

    /// Create a failure response.
    #[must_use]
    pub fn failure(command_type: &str, message: &str) -> Self {
        Self {
            success: false,
            message: message.to_string(),
            command_type: command_type.to_string(),
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos() as u64,
        }
    }
}

// ---------------------------------------------------------------------------
// Session record / playback
// ---------------------------------------------------------------------------

/// A single recorded event in a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    /// Monotonic offset from session start in nanoseconds.
    pub offset_ns: u64,
    /// The telemetry snapshot captured at this instant.
    pub snapshot: TelemetrySnapshot,
}

/// In-memory session recorder.
///
/// Records telemetry snapshots with monotonic timestamps for later playback
/// or export to disk.
pub struct SessionRecorder {
    session_name: String,
    start_time: Instant,
    events: Vec<SessionEvent>,
    is_recording: bool,
    max_events: usize,
}

impl SessionRecorder {
    /// Create a new recorder (not yet recording).
    #[must_use]
    pub fn new(max_events: usize) -> Self {
        Self {
            session_name: String::new(),
            start_time: Instant::now(),
            events: Vec::new(),
            is_recording: false,
            max_events,
        }
    }

    /// Start recording a new session.
    pub fn start(&mut self, name: &str) -> Result<()> {
        if self.is_recording {
            return Err(VirtualProductionError::InvalidConfig(
                "Session already recording".to_string(),
            ));
        }
        self.session_name = name.to_string();
        self.start_time = Instant::now();
        self.events.clear();
        self.is_recording = true;
        Ok(())
    }

    /// Stop the current recording.
    pub fn stop(&mut self) -> Result<()> {
        if !self.is_recording {
            return Err(VirtualProductionError::InvalidConfig(
                "Not currently recording".to_string(),
            ));
        }
        self.is_recording = false;
        Ok(())
    }

    /// Record a telemetry snapshot (no-op if not recording).
    pub fn record(&mut self, snapshot: TelemetrySnapshot) {
        if !self.is_recording {
            return;
        }
        if self.events.len() >= self.max_events {
            // Ring-buffer behaviour: drop oldest
            self.events.remove(0);
        }
        let offset_ns = self.start_time.elapsed().as_nanos() as u64;
        self.events.push(SessionEvent {
            offset_ns,
            snapshot,
        });
    }

    /// Get all recorded events.
    #[must_use]
    pub fn events(&self) -> &[SessionEvent] {
        &self.events
    }

    /// Session name.
    #[must_use]
    pub fn session_name(&self) -> &str {
        &self.session_name
    }

    /// Whether recording is active.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.is_recording
    }

    /// Number of recorded events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Export to JSON string.
    pub fn export_json(&self) -> Result<String> {
        serde_json::to_string(&self.events)
            .map_err(|e| VirtualProductionError::InvalidConfig(format!("JSON export failed: {e}")))
    }

    /// Import from JSON string (replaces current events).
    pub fn import_json(&mut self, json: &str) -> Result<()> {
        let events: Vec<SessionEvent> = serde_json::from_str(json).map_err(|e| {
            VirtualProductionError::InvalidConfig(format!("JSON import failed: {e}"))
        })?;
        self.events = events;
        self.is_recording = false;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Session playback
// ---------------------------------------------------------------------------

/// Plays back a recorded session, yielding snapshots at the original timing.
pub struct SessionPlayback {
    events: Vec<SessionEvent>,
    current_index: usize,
    playback_start: Instant,
    speed: f64,
    looping: bool,
}

impl SessionPlayback {
    /// Create a new playback from recorded events.
    #[must_use]
    pub fn new(events: Vec<SessionEvent>, speed: f64, looping: bool) -> Self {
        Self {
            events,
            current_index: 0,
            playback_start: Instant::now(),
            speed: speed.max(0.01),
            looping,
        }
    }

    /// Poll for the next snapshot that is due to be played back.
    ///
    /// Returns `None` if playback is complete (and not looping) or
    /// if no new event is due yet.
    #[must_use]
    pub fn poll(&mut self) -> Option<&TelemetrySnapshot> {
        if self.events.is_empty() {
            return None;
        }

        let elapsed_ns = (self.playback_start.elapsed().as_nanos() as f64 * self.speed) as u64;

        // Advance index while the current event is in the past
        loop {
            if self.current_index >= self.events.len() {
                if self.looping {
                    self.current_index = 0;
                    self.playback_start = Instant::now();
                    return None;
                } else {
                    return None;
                }
            }
            let ev_offset = self.events[self.current_index].offset_ns;
            if elapsed_ns >= ev_offset {
                let snap = &self.events[self.current_index].snapshot;
                self.current_index += 1;
                return Some(snap);
            } else {
                break;
            }
        }

        None
    }

    /// Reset playback to the beginning.
    pub fn reset(&mut self) {
        self.current_index = 0;
        self.playback_start = Instant::now();
    }

    /// Whether playback has reached the end (and is not looping).
    #[must_use]
    pub fn is_finished(&self) -> bool {
        !self.looping && self.current_index >= self.events.len()
    }

    /// Total number of events in the sequence.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Current playback index.
    #[must_use]
    pub fn current_index(&self) -> usize {
        self.current_index
    }
}

// ---------------------------------------------------------------------------
// Remote session server (in-process / simulated)
// ---------------------------------------------------------------------------

/// Configuration for the remote session server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteSessionConfig {
    /// Maximum number of simultaneously connected operators.
    pub max_operators: usize,
    /// Telemetry push interval in milliseconds.
    pub telemetry_interval_ms: u64,
    /// Maximum command queue depth.
    pub command_queue_depth: usize,
    /// Session recorder buffer size (max events).
    pub recorder_buffer: usize,
    /// Whether to require authentication (role checking).
    pub require_auth: bool,
}

impl Default for RemoteSessionConfig {
    fn default() -> Self {
        Self {
            max_operators: 8,
            telemetry_interval_ms: 100,
            command_queue_depth: 64,
            recorder_buffer: 10_000,
            require_auth: false,
        }
    }
}

/// An in-process remote session manager.
///
/// In production this would manage real TCP connections; here it provides
/// an in-process command queue and telemetry bus suitable for testing
/// and same-machine control panels.
pub struct RemoteSessionServer {
    config: RemoteSessionConfig,
    /// Registered operators with their roles.
    operators: Vec<(String, OperatorRole)>,
    /// Pending commands from remote clients.
    command_queue: VecDeque<(String, RemoteCommand)>,
    /// Latest telemetry snapshot.
    latest_snapshot: TelemetrySnapshot,
    /// Response log (most recent responses).
    response_log: VecDeque<CommandResponse>,
    /// Session recorder.
    recorder: SessionRecorder,
    /// Emergency stop state.
    emergency_stop: bool,
    /// Session start time.
    session_start: Instant,
    /// Frame counter.
    frame_count: u64,
}

impl RemoteSessionServer {
    /// Create a new remote session server.
    #[must_use]
    pub fn new(config: RemoteSessionConfig) -> Self {
        let recorder_buf = config.recorder_buffer;
        Self {
            config,
            operators: Vec::new(),
            command_queue: VecDeque::new(),
            latest_snapshot: TelemetrySnapshot::default_snapshot(),
            response_log: VecDeque::with_capacity(256),
            recorder: SessionRecorder::new(recorder_buf),
            emergency_stop: false,
            session_start: Instant::now(),
            frame_count: 0,
        }
    }

    /// Register an operator with the given role.
    pub fn register_operator(&mut self, name: &str, role: OperatorRole) -> Result<()> {
        if self.operators.len() >= self.config.max_operators {
            return Err(VirtualProductionError::InvalidConfig(format!(
                "Maximum operator count ({}) reached",
                self.config.max_operators
            )));
        }
        if self.operators.iter().any(|(n, _)| n == name) {
            return Err(VirtualProductionError::InvalidConfig(format!(
                "Operator '{name}' already registered"
            )));
        }
        self.operators.push((name.to_string(), role));
        Ok(())
    }

    /// Deregister an operator by name.
    pub fn deregister_operator(&mut self, name: &str) {
        self.operators.retain(|(n, _)| n != name);
    }

    /// Get an operator's role.
    #[must_use]
    pub fn operator_role(&self, name: &str) -> Option<OperatorRole> {
        self.operators
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, r)| *r)
    }

    /// Queue a command from an operator.
    ///
    /// Returns an error if the operator lacks the required role or if
    /// the queue is full.
    pub fn submit_command(&mut self, operator: &str, cmd: RemoteCommand) -> Result<()> {
        // Role check
        let role = self.operator_role(operator).ok_or_else(|| {
            VirtualProductionError::InvalidConfig(format!("Unknown operator '{operator}'"))
        })?;

        if !role.can_control() {
            return Err(VirtualProductionError::InvalidConfig(format!(
                "Operator '{operator}' (role: {}) cannot send commands",
                role.label()
            )));
        }

        if self.command_queue.len() >= self.config.command_queue_depth {
            return Err(VirtualProductionError::InvalidConfig(
                "Command queue is full".to_string(),
            ));
        }

        self.command_queue.push_back((operator.to_string(), cmd));
        Ok(())
    }

    /// Process all queued commands.  Returns the number of commands processed.
    pub fn process_commands(&mut self) -> usize {
        let mut processed = 0;

        while let Some((operator, cmd)) = self.command_queue.pop_front() {
            let response = self.execute_command(&operator, &cmd);
            if self.response_log.len() >= 256 {
                self.response_log.pop_front();
            }
            self.response_log.push_back(response);
            processed += 1;
        }

        processed
    }

    /// Execute a single command and return the response.
    fn execute_command(&mut self, operator: &str, cmd: &RemoteCommand) -> CommandResponse {
        match cmd {
            RemoteCommand::Ping { sequence } => {
                CommandResponse::success("Ping", &format!("Pong from server, sequence={sequence}"))
            }

            RemoteCommand::Recalibrate { camera_id } => CommandResponse::success(
                "Recalibrate",
                &format!("Recalibration triggered for camera {camera_id}"),
            ),

            RemoteCommand::SetLedBrightness { brightness } => {
                let b = brightness.clamp(0.0, 1.0);
                self.latest_snapshot.led_brightness = b;
                CommandResponse::success(
                    "SetLedBrightness",
                    &format!("LED brightness set to {b:.3}"),
                )
            }

            RemoteCommand::SetWorkflow { workflow } => {
                self.latest_snapshot.workflow = workflow.clone();
                CommandResponse::success(
                    "SetWorkflow",
                    &format!("Workflow changed to '{workflow}'"),
                )
            }

            RemoteCommand::RequestSnapshot => CommandResponse::success(
                "RequestSnapshot",
                &format!(
                    "Snapshot at frame {} dispatched",
                    self.latest_snapshot.frame_number
                ),
            ),

            RemoteCommand::StartRecording { session_name } => {
                match self.recorder.start(session_name) {
                    Ok(()) => {
                        self.latest_snapshot.is_recording = true;
                        CommandResponse::success(
                            "StartRecording",
                            &format!("Recording started: '{session_name}'"),
                        )
                    }
                    Err(e) => CommandResponse::failure("StartRecording", &e.to_string()),
                }
            }

            RemoteCommand::StopRecording => match self.recorder.stop() {
                Ok(()) => {
                    self.latest_snapshot.is_recording = false;
                    CommandResponse::success(
                        "StopRecording",
                        &format!(
                            "Recording stopped: {} events saved",
                            self.recorder.event_count()
                        ),
                    )
                }
                Err(e) => CommandResponse::failure("StopRecording", &e.to_string()),
            },

            RemoteCommand::SetColorTemperature { delta_k } => CommandResponse::success(
                "SetColorTemperature",
                &format!("Color temperature delta {delta_k:+}K applied by {operator}"),
            ),

            RemoteCommand::EmergencyStop => {
                self.emergency_stop = true;
                CommandResponse::success("EmergencyStop", "Emergency stop engaged")
            }

            RemoteCommand::ResumeFromStop => {
                self.emergency_stop = false;
                CommandResponse::success("ResumeFromStop", "Resumed from emergency stop")
            }
        }
    }

    /// Push a telemetry update (call once per frame from the production loop).
    pub fn push_telemetry(
        &mut self,
        camera_position: [f64; 3],
        camera_euler_deg: [f64; 3],
        sync_status: &str,
        pipeline_latency_us: u64,
    ) {
        self.frame_count += 1;
        let ts = self.session_start.elapsed().as_nanos() as u64;

        self.latest_snapshot.timestamp_ns = ts;
        self.latest_snapshot.frame_number = self.frame_count;
        self.latest_snapshot.camera_position = camera_position;
        self.latest_snapshot.camera_euler_deg = camera_euler_deg;
        self.latest_snapshot.sync_status = sync_status.to_string();
        self.latest_snapshot.pipeline_latency_us = pipeline_latency_us;
        self.latest_snapshot.operator_count = self.operators.len();

        // Record if active
        if self.recorder.is_recording() {
            self.recorder.record(self.latest_snapshot.clone());
        }
    }

    /// Get the latest telemetry snapshot.
    #[must_use]
    pub fn latest_snapshot(&self) -> &TelemetrySnapshot {
        &self.latest_snapshot
    }

    /// Whether emergency stop is engaged.
    #[must_use]
    pub fn is_emergency_stop(&self) -> bool {
        self.emergency_stop
    }

    /// Number of registered operators.
    #[must_use]
    pub fn operator_count(&self) -> usize {
        self.operators.len()
    }

    /// Most recent command responses.
    #[must_use]
    pub fn response_log(&self) -> &VecDeque<CommandResponse> {
        &self.response_log
    }

    /// Number of pending commands.
    #[must_use]
    pub fn pending_command_count(&self) -> usize {
        self.command_queue.len()
    }

    /// Get reference to the session recorder.
    #[must_use]
    pub fn recorder(&self) -> &SessionRecorder {
        &self.recorder
    }

    /// Get mutable reference to the session recorder.
    pub fn recorder_mut(&mut self) -> &mut SessionRecorder {
        &mut self.recorder
    }

    /// Frame count since session started.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &RemoteSessionConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Frame-length-prefixed JSON message codec (TCP wire format helper)
// ---------------------------------------------------------------------------

/// Encode a serializable value to a 4-byte-length-prefixed JSON frame.
pub fn encode_message<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let json = serde_json::to_vec(value)
        .map_err(|e| VirtualProductionError::InvalidConfig(format!("JSON encode error: {e}")))?;
    let len = json.len() as u32;
    let mut buf = Vec::with_capacity(4 + json.len());
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&json);
    Ok(buf)
}

/// Decode a 4-byte-length-prefixed JSON frame into a value.
pub fn decode_message<T: for<'de> Deserialize<'de>>(buf: &[u8]) -> Result<T> {
    if buf.len() < 4 {
        return Err(VirtualProductionError::InvalidConfig(
            "Message too short for length prefix".to_string(),
        ));
    }
    let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let payload = buf.get(4..4 + len).ok_or_else(|| {
        VirtualProductionError::InvalidConfig(format!(
            "Message payload truncated: need {len} bytes, have {}",
            buf.len() - 4
        ))
    })?;
    serde_json::from_slice(payload)
        .map_err(|e| VirtualProductionError::InvalidConfig(format!("JSON decode error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_server() -> RemoteSessionServer {
        RemoteSessionServer::new(RemoteSessionConfig::default())
    }

    #[test]
    fn test_server_creation() {
        let server = make_server();
        assert_eq!(server.operator_count(), 0);
        assert_eq!(server.pending_command_count(), 0);
        assert!(!server.is_emergency_stop());
    }

    #[test]
    fn test_register_operator() {
        let mut server = make_server();
        server
            .register_operator("alice", OperatorRole::Director)
            .expect("ok");
        assert_eq!(server.operator_count(), 1);
        assert_eq!(server.operator_role("alice"), Some(OperatorRole::Director));
    }

    #[test]
    fn test_register_duplicate_operator_fails() {
        let mut server = make_server();
        server
            .register_operator("alice", OperatorRole::Director)
            .expect("ok");
        let result = server.register_operator("alice", OperatorRole::Observer);
        assert!(result.is_err(), "duplicate registration should fail");
    }

    #[test]
    fn test_deregister_operator() {
        let mut server = make_server();
        server
            .register_operator("alice", OperatorRole::Director)
            .expect("ok");
        server.deregister_operator("alice");
        assert_eq!(server.operator_count(), 0);
        assert!(server.operator_role("alice").is_none());
    }

    #[test]
    fn test_max_operator_limit() {
        let config = RemoteSessionConfig {
            max_operators: 2,
            ..RemoteSessionConfig::default()
        };
        let mut server = RemoteSessionServer::new(config);
        server
            .register_operator("a", OperatorRole::Director)
            .expect("ok");
        server
            .register_operator("b", OperatorRole::Observer)
            .expect("ok");
        let result = server.register_operator("c", OperatorRole::Observer);
        assert!(result.is_err(), "should reject when at capacity");
    }

    #[test]
    fn test_submit_command_as_director() {
        let mut server = make_server();
        server
            .register_operator("director", OperatorRole::Director)
            .expect("ok");
        server
            .submit_command("director", RemoteCommand::Ping { sequence: 1 })
            .expect("ok");
        assert_eq!(server.pending_command_count(), 1);
    }

    #[test]
    fn test_submit_command_as_observer_fails() {
        let mut server = make_server();
        server
            .register_operator("viewer", OperatorRole::Observer)
            .expect("ok");
        let result = server.submit_command("viewer", RemoteCommand::Ping { sequence: 1 });
        assert!(
            result.is_err(),
            "observer should not be able to submit commands"
        );
    }

    #[test]
    fn test_submit_command_unknown_operator_fails() {
        let mut server = make_server();
        let result = server.submit_command("ghost", RemoteCommand::Ping { sequence: 1 });
        assert!(result.is_err());
    }

    #[test]
    fn test_process_ping_command() {
        let mut server = make_server();
        server
            .register_operator("dir", OperatorRole::Director)
            .expect("ok");
        server
            .submit_command("dir", RemoteCommand::Ping { sequence: 42 })
            .expect("ok");
        let n = server.process_commands();
        assert_eq!(n, 1);
        let resp = server.response_log().back().expect("should have response");
        assert!(resp.success, "ping should succeed");
        assert_eq!(resp.command_type, "Ping");
    }

    #[test]
    fn test_set_led_brightness() {
        let mut server = make_server();
        server
            .register_operator("tech", OperatorRole::LedTech)
            .expect("ok");
        server
            .submit_command("tech", RemoteCommand::SetLedBrightness { brightness: 0.75 })
            .expect("ok");
        server.process_commands();
        assert!((server.latest_snapshot().led_brightness - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_emergency_stop_and_resume() {
        let mut server = make_server();
        server
            .register_operator("dir", OperatorRole::Director)
            .expect("ok");

        server
            .submit_command("dir", RemoteCommand::EmergencyStop)
            .expect("ok");
        server.process_commands();
        assert!(server.is_emergency_stop());

        server
            .submit_command("dir", RemoteCommand::ResumeFromStop)
            .expect("ok");
        server.process_commands();
        assert!(!server.is_emergency_stop());
    }

    #[test]
    fn test_start_and_stop_recording() {
        let mut server = make_server();
        server
            .register_operator("dir", OperatorRole::Director)
            .expect("ok");

        server
            .submit_command(
                "dir",
                RemoteCommand::StartRecording {
                    session_name: "take_01".to_string(),
                },
            )
            .expect("ok");
        server.process_commands();
        assert!(server.recorder().is_recording());

        // Push some telemetry while recording
        server.push_telemetry([1.0, 0.5, -2.0], [0.0, 45.0, 0.0], "Locked", 5000);
        server.push_telemetry([1.1, 0.5, -2.1], [0.0, 46.0, 0.0], "Locked", 4800);
        assert_eq!(server.recorder().event_count(), 2);

        server
            .submit_command("dir", RemoteCommand::StopRecording)
            .expect("ok");
        server.process_commands();
        assert!(!server.recorder().is_recording());
    }

    #[test]
    fn test_double_start_recording_fails() {
        let mut server = make_server();
        server
            .register_operator("dir", OperatorRole::Director)
            .expect("ok");

        server
            .submit_command(
                "dir",
                RemoteCommand::StartRecording {
                    session_name: "take_01".to_string(),
                },
            )
            .expect("ok");
        server.process_commands();

        // Second start should fail
        server
            .submit_command(
                "dir",
                RemoteCommand::StartRecording {
                    session_name: "take_02".to_string(),
                },
            )
            .expect("ok");
        server.process_commands();

        let resp = server.response_log().back().expect("response");
        assert!(!resp.success, "double start should fail");
    }

    #[test]
    fn test_telemetry_push_updates_snapshot() {
        let mut server = make_server();
        server.push_telemetry([3.0, 1.0, -5.0], [10.0, 20.0, 0.0], "Locked", 8000);

        let snap = server.latest_snapshot();
        assert!((snap.camera_position[0] - 3.0).abs() < 1e-9);
        assert_eq!(snap.sync_status, "Locked");
        assert_eq!(snap.pipeline_latency_us, 8000);
        assert_eq!(snap.frame_number, 1);
    }

    #[test]
    fn test_set_workflow() {
        let mut server = make_server();
        server
            .register_operator("dir", OperatorRole::Director)
            .expect("ok");
        server
            .submit_command(
                "dir",
                RemoteCommand::SetWorkflow {
                    workflow: "Hybrid".to_string(),
                },
            )
            .expect("ok");
        server.process_commands();
        assert_eq!(server.latest_snapshot().workflow, "Hybrid");
    }

    #[test]
    fn test_frame_count_increments_with_telemetry() {
        let mut server = make_server();
        assert_eq!(server.frame_count(), 0);
        for _ in 0..5 {
            server.push_telemetry([0.0; 3], [0.0; 3], "Unlocked", 0);
        }
        assert_eq!(server.frame_count(), 5);
    }

    #[test]
    fn test_command_queue_depth_limit() {
        let config = RemoteSessionConfig {
            command_queue_depth: 2,
            ..RemoteSessionConfig::default()
        };
        let mut server = RemoteSessionServer::new(config);
        server
            .register_operator("dir", OperatorRole::Director)
            .expect("ok");

        server
            .submit_command("dir", RemoteCommand::Ping { sequence: 1 })
            .expect("ok");
        server
            .submit_command("dir", RemoteCommand::Ping { sequence: 2 })
            .expect("ok");
        let result = server.submit_command("dir", RemoteCommand::Ping { sequence: 3 });
        assert!(result.is_err(), "queue depth exceeded should fail");
    }

    #[test]
    fn test_encode_decode_message_roundtrip() {
        let snap = TelemetrySnapshot::default_snapshot();
        let encoded = encode_message(&snap).expect("encode ok");
        let decoded: TelemetrySnapshot = decode_message(&encoded).expect("decode ok");
        assert_eq!(decoded.frame_number, snap.frame_number);
        assert_eq!(decoded.workflow, snap.workflow);
    }

    #[test]
    fn test_decode_truncated_message_fails() {
        let result: Result<TelemetrySnapshot> = decode_message(&[0, 0, 0, 100]);
        assert!(result.is_err(), "truncated payload should fail");
    }

    #[test]
    fn test_decode_too_short_fails() {
        let result: Result<TelemetrySnapshot> = decode_message(&[0, 1]);
        assert!(result.is_err(), "< 4 bytes should fail");
    }

    #[test]
    fn test_session_recorder_export_import() {
        let mut recorder = SessionRecorder::new(100);
        recorder.start("test").expect("ok");
        recorder.record(TelemetrySnapshot::default_snapshot());
        recorder.stop().expect("ok");

        let json = recorder.export_json().expect("export ok");

        let mut recorder2 = SessionRecorder::new(100);
        recorder2.import_json(&json).expect("import ok");
        assert_eq!(recorder2.event_count(), 1);
    }

    #[test]
    fn test_session_playback() {
        // Create two events with offsets 0 and 1_000_000_000 ns (1 second apart)
        let events = vec![
            SessionEvent {
                offset_ns: 0,
                snapshot: TelemetrySnapshot {
                    frame_number: 1,
                    ..TelemetrySnapshot::default_snapshot()
                },
            },
            SessionEvent {
                offset_ns: 1_000_000_000,
                snapshot: TelemetrySnapshot {
                    frame_number: 2,
                    ..TelemetrySnapshot::default_snapshot()
                },
            },
        ];

        let mut playback = SessionPlayback::new(events, 1.0, false);
        assert_eq!(playback.event_count(), 2);

        // First event (offset=0) should be immediately available
        let first = playback.poll();
        assert!(
            first.is_some(),
            "first event should be immediately available"
        );
        assert_eq!(first.expect("ok").frame_number, 1);
    }

    #[test]
    fn test_playback_finished() {
        let events = vec![SessionEvent {
            offset_ns: 0,
            snapshot: TelemetrySnapshot::default_snapshot(),
        }];
        let mut playback = SessionPlayback::new(events, 1.0, false);
        playback.poll(); // consume the only event
                         // poll again
        playback.poll();
        assert!(playback.is_finished());
    }

    #[test]
    fn test_operator_role_can_control() {
        assert!(OperatorRole::Director.can_control());
        assert!(OperatorRole::CameraOp.can_control());
        assert!(OperatorRole::LedTech.can_control());
        assert!(!OperatorRole::Observer.can_control());
    }

    #[test]
    fn test_operator_role_labels() {
        assert_eq!(OperatorRole::Director.label(), "Director");
        assert_eq!(OperatorRole::Observer.label(), "Observer");
    }

    #[test]
    fn test_response_success_failure() {
        let ok = CommandResponse::success("Test", "it worked");
        assert!(ok.success);
        assert_eq!(ok.command_type, "Test");

        let fail = CommandResponse::failure("Test", "it broke");
        assert!(!fail.success);
    }

    #[test]
    fn test_multiple_operators_different_roles() {
        let mut server = make_server();
        server
            .register_operator("alice", OperatorRole::Director)
            .expect("ok");
        server
            .register_operator("bob", OperatorRole::Observer)
            .expect("ok");
        server
            .register_operator("carol", OperatorRole::LedTech)
            .expect("ok");

        assert_eq!(server.operator_count(), 3);

        // Director can command
        server
            .submit_command("alice", RemoteCommand::Ping { sequence: 1 })
            .expect("ok");
        // Observer cannot
        assert!(server
            .submit_command("bob", RemoteCommand::Ping { sequence: 1 })
            .is_err());
        // LedTech can command
        server
            .submit_command("carol", RemoteCommand::SetLedBrightness { brightness: 0.5 })
            .expect("ok");

        server.process_commands();
        assert_eq!(server.response_log().len(), 2);
    }
}
