//! PTP port management.

use super::bmca::PortState as BmcaPortState;
use super::{DelayMechanism, PortIdentity};
use std::time::Duration;

/// PTP port configuration.
#[derive(Debug, Clone)]
pub struct PortConfig {
    /// Port number (1-based)
    pub port_number: u16,
    /// Delay mechanism
    pub delay_mechanism: DelayMechanism,
    /// Announce receipt timeout (number of intervals)
    pub announce_receipt_timeout: u8,
    /// Sync receipt timeout
    pub sync_receipt_timeout: u8,
    /// Delay request interval (log2 seconds)
    pub delay_request_interval: i8,
    /// Announce interval (log2 seconds)
    pub announce_interval: i8,
    /// Sync interval (log2 seconds)
    pub sync_interval: i8,
}

impl Default for PortConfig {
    fn default() -> Self {
        Self {
            port_number: 1,
            delay_mechanism: DelayMechanism::E2E,
            announce_receipt_timeout: 3,
            sync_receipt_timeout: 3,
            delay_request_interval: 0, // 1 second
            announce_interval: 1,      // 2 seconds
            sync_interval: 0,          // 1 second
        }
    }
}

impl PortConfig {
    /// Get announce interval as duration.
    #[must_use]
    pub fn announce_interval_duration(&self) -> Duration {
        interval_to_duration(self.announce_interval)
    }

    /// Get sync interval as duration.
    #[must_use]
    pub fn sync_interval_duration(&self) -> Duration {
        interval_to_duration(self.sync_interval)
    }

    /// Get delay request interval as duration.
    #[must_use]
    pub fn delay_request_interval_duration(&self) -> Duration {
        interval_to_duration(self.delay_request_interval)
    }

    /// Get announce receipt timeout as duration.
    #[must_use]
    pub fn announce_timeout_duration(&self) -> Duration {
        self.announce_interval_duration()
            .mul_f64(f64::from(self.announce_receipt_timeout))
    }
}

/// Convert log2 interval to duration.
fn interval_to_duration(log_interval: i8) -> Duration {
    if log_interval >= 0 {
        Duration::from_secs(1 << log_interval)
    } else {
        Duration::from_millis(1000 >> (-log_interval))
    }
}

/// PTP port runtime state.
#[derive(Debug)]
pub struct PtpPortState {
    /// Port identity
    pub identity: PortIdentity,
    /// Current state
    pub state: BmcaPortState,
    /// Configuration
    pub config: PortConfig,
    /// Last sync sequence ID
    pub last_sync_seq: Option<u16>,
    /// Last announce sequence ID
    pub last_announce_seq: Option<u16>,
}

impl PtpPortState {
    /// Create a new port state.
    #[must_use]
    pub fn new(identity: PortIdentity, config: PortConfig) -> Self {
        Self {
            identity,
            state: BmcaPortState::Initializing,
            config,
            last_sync_seq: None,
            last_announce_seq: None,
        }
    }

    /// Update state.
    pub fn set_state(&mut self, state: BmcaPortState) {
        self.state = state;
    }

    /// Check if port is master.
    #[must_use]
    pub fn is_master(&self) -> bool {
        self.state == BmcaPortState::Master
    }

    /// Check if port is slave.
    #[must_use]
    pub fn is_slave(&self) -> bool {
        self.state == BmcaPortState::Slave
    }
}

// ---------------------------------------------------------------------------
// Batched PTP message processing
// ---------------------------------------------------------------------------

/// A raw PTP message envelope, as received from the network.
#[derive(Debug, Clone)]
pub struct RawPtpMessage {
    /// The raw bytes of the PTP message (common header + body).
    pub data: Vec<u8>,
    /// Reception timestamp in nanoseconds since UNIX epoch (software
    /// timestamp; hardware timestamp should be used when available).
    pub recv_timestamp_ns: u64,
}

impl RawPtpMessage {
    /// Creates a new raw message.
    #[must_use]
    pub fn new(data: Vec<u8>, recv_timestamp_ns: u64) -> Self {
        Self {
            data,
            recv_timestamp_ns,
        }
    }
}

/// Classification of a raw PTP message by its type field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtpMessageClass {
    /// Sync / Follow_Up pair (event + general).
    SyncPair,
    /// Delay_Req / Delay_Resp pair.
    DelayReqResp,
    /// Announce message.
    Announce,
    /// Management message.
    Management,
    /// Other / unknown.
    Other,
}

impl PtpMessageClass {
    /// Classify by the lower nibble of the first byte (messageType field).
    #[must_use]
    pub fn from_first_byte(byte: u8) -> Self {
        match byte & 0x0F {
            0x0 | 0x8 => Self::SyncPair,     // Sync or Follow_Up
            0x1 | 0x9 => Self::DelayReqResp, // Delay_Req or Delay_Resp
            0xB => Self::Announce,
            0xD => Self::Management,
            _ => Self::Other,
        }
    }
}

/// Result of processing a single message in a batch.
#[derive(Debug, Clone)]
pub struct BatchProcessResult {
    /// The class of message that was processed.
    pub class: PtpMessageClass,
    /// Sequence ID extracted from bytes 30–31 (0 if parsing failed).
    pub sequence_id: u16,
    /// Whether the message passed basic validation.
    pub valid: bool,
    /// Optional error description for invalid messages.
    pub error: Option<String>,
}

/// Batch processor for bursts of incoming PTP messages.
///
/// When PTP Sync / Follow_Up messages arrive in rapid succession (e.g. at
/// 8 msg/s or faster), grouping them into a batch and processing them
/// sequentially reduces per-message overhead and enables in-order
/// Sync+Follow_Up pairing.
///
/// # Usage
/// ```rust,ignore
/// let mut processor = PtpBatchProcessor::new(32);
/// processor.enqueue(RawPtpMessage::new(bytes, timestamp_ns));
/// // ... enqueue more ...
/// let results = processor.process_all();
/// ```
#[derive(Debug)]
pub struct PtpBatchProcessor {
    /// The pending message queue.
    queue: std::collections::VecDeque<RawPtpMessage>,
    /// Maximum number of messages to accept before the oldest is dropped.
    max_queue_depth: usize,
    /// Number of messages dropped due to queue overflow.
    dropped_count: u64,
    /// Total messages processed.
    processed_count: u64,
}

impl PtpBatchProcessor {
    /// Creates a new batch processor with the specified maximum queue depth.
    ///
    /// A depth of 32–128 is typical; bursts beyond `max_queue_depth` will
    /// drop the oldest messages.
    #[must_use]
    pub fn new(max_queue_depth: usize) -> Self {
        Self {
            queue: std::collections::VecDeque::with_capacity(max_queue_depth.min(64)),
            max_queue_depth: max_queue_depth.max(1),
            dropped_count: 0,
            processed_count: 0,
        }
    }

    /// Enqueues a raw PTP message for batch processing.
    ///
    /// If the queue is at capacity, the oldest message is dropped and
    /// `dropped_count` is incremented.
    pub fn enqueue(&mut self, msg: RawPtpMessage) {
        if self.queue.len() >= self.max_queue_depth {
            self.queue.pop_front();
            self.dropped_count += 1;
        }
        self.queue.push_back(msg);
    }

    /// Returns the number of messages currently in the queue.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// Returns the total number of dropped messages.
    #[must_use]
    pub fn dropped_count(&self) -> u64 {
        self.dropped_count
    }

    /// Returns the total number of successfully processed messages.
    #[must_use]
    pub fn processed_count(&self) -> u64 {
        self.processed_count
    }

    /// Processes all queued messages in FIFO order, returning a
    /// [`BatchProcessResult`] for each one.
    ///
    /// The queue is drained completely by this call.
    pub fn process_all(&mut self) -> Vec<BatchProcessResult> {
        let count = self.queue.len();
        let mut results = Vec::with_capacity(count);

        while let Some(msg) = self.queue.pop_front() {
            let result = Self::process_single(&msg);
            if result.valid {
                self.processed_count += 1;
            }
            results.push(result);
        }
        results
    }

    /// Processes up to `limit` messages from the queue.
    pub fn process_up_to(&mut self, limit: usize) -> Vec<BatchProcessResult> {
        let take = self.queue.len().min(limit);
        let mut results = Vec::with_capacity(take);
        for _ in 0..take {
            if let Some(msg) = self.queue.pop_front() {
                let result = Self::process_single(&msg);
                if result.valid {
                    self.processed_count += 1;
                }
                results.push(result);
            }
        }
        results
    }

    /// Validates and classifies a single raw message.
    fn process_single(msg: &RawPtpMessage) -> BatchProcessResult {
        // Minimum PTP message size is 34 bytes (common header only).
        if msg.data.len() < 34 {
            return BatchProcessResult {
                class: PtpMessageClass::Other,
                sequence_id: 0,
                valid: false,
                error: Some(format!(
                    "Message too short: {} bytes (minimum 34)",
                    msg.data.len()
                )),
            };
        }

        let first_byte = msg.data[0];
        let class = PtpMessageClass::from_first_byte(first_byte);

        // Extract sequence ID from bytes 30–31 (big-endian).
        let sequence_id = u16::from_be_bytes([msg.data[30], msg.data[31]]);

        // Validate PTP version (upper nibble of byte 0, or byte 1 in some
        // implementations). IEEE 1588v2 version = 0x02.
        let version = (msg.data[1]) & 0x0F;
        if version != 2 {
            return BatchProcessResult {
                class,
                sequence_id,
                valid: false,
                error: Some(format!("Unsupported PTP version: {version}")),
            };
        }

        // Check declared message length against actual data length.
        let declared_len = u16::from_be_bytes([msg.data[2], msg.data[3]]) as usize;
        if declared_len > msg.data.len() {
            return BatchProcessResult {
                class,
                sequence_id,
                valid: false,
                error: Some(format!(
                    "Declared length {declared_len} > actual length {}",
                    msg.data.len()
                )),
            };
        }

        BatchProcessResult {
            class,
            sequence_id,
            valid: true,
            error: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Sync / Follow_Up pairing helper
// ---------------------------------------------------------------------------

/// A matched Sync + Follow_Up pair for two-step clock processing.
#[derive(Debug, Clone)]
pub struct SyncPair {
    /// Sequence ID (must match between Sync and Follow_Up).
    pub sequence_id: u16,
    /// Reception timestamp of the Sync message.
    pub sync_recv_ns: u64,
    /// Precise origin timestamp from the Follow_Up (nanoseconds since epoch).
    pub precise_origin_ns: u64,
}

/// Buffers incoming two-step Sync and Follow_Up messages and emits
/// [`SyncPair`]s when both halves of a pair are available.
#[derive(Debug, Default)]
pub struct SyncFollowUpPairer {
    /// Buffered Sync messages awaiting their Follow_Up: seq_id → recv_ns.
    pending_syncs: std::collections::HashMap<u16, u64>,
}

impl SyncFollowUpPairer {
    /// Creates a new pairer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a Sync message reception.
    ///
    /// `sequence_id`   — sequence number from the Sync header.
    /// `recv_ns`       — reception timestamp of the Sync in nanoseconds.
    pub fn record_sync(&mut self, sequence_id: u16, recv_ns: u64) {
        self.pending_syncs.insert(sequence_id, recv_ns);
    }

    /// Records a Follow_Up message and tries to match it with a pending Sync.
    ///
    /// `sequence_id`        — must match the corresponding Sync.
    /// `precise_origin_ns`  — precise origin timestamp from the Follow_Up body.
    ///
    /// Returns `Some(SyncPair)` if a matching Sync was found, `None` otherwise.
    pub fn record_follow_up(
        &mut self,
        sequence_id: u16,
        precise_origin_ns: u64,
    ) -> Option<SyncPair> {
        let sync_recv_ns = self.pending_syncs.remove(&sequence_id)?;
        Some(SyncPair {
            sequence_id,
            sync_recv_ns,
            precise_origin_ns,
        })
    }

    /// Returns the number of Sync messages awaiting a Follow_Up.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending_syncs.len()
    }

    /// Removes all pending Syncs (e.g. after a state transition).
    pub fn clear(&mut self) {
        self.pending_syncs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptp::ClockIdentity;

    // -----------------------------------------------------------------------
    // Original tests (unchanged)
    // -----------------------------------------------------------------------

    #[test]
    fn test_interval_to_duration() {
        assert_eq!(interval_to_duration(0), Duration::from_secs(1));
        assert_eq!(interval_to_duration(1), Duration::from_secs(2));
        assert_eq!(interval_to_duration(2), Duration::from_secs(4));
        assert_eq!(interval_to_duration(-1), Duration::from_millis(500));
        assert_eq!(interval_to_duration(-2), Duration::from_millis(250));
    }

    #[test]
    fn test_port_config_default() {
        let config = PortConfig::default();
        assert_eq!(config.port_number, 1);
        assert_eq!(config.delay_mechanism, DelayMechanism::E2E);
        assert_eq!(config.sync_interval_duration(), Duration::from_secs(1));
    }

    #[test]
    fn test_port_state_creation() {
        let clock_id = ClockIdentity::random();
        let port_id = PortIdentity::new(clock_id, 1);
        let config = PortConfig::default();
        let state = PtpPortState::new(port_id, config);

        assert_eq!(state.identity, port_id);
        assert!(!state.is_master());
        assert!(!state.is_slave());
    }

    // -----------------------------------------------------------------------
    // PtpMessageClass tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_message_class_sync() {
        assert_eq!(
            PtpMessageClass::from_first_byte(0x00),
            PtpMessageClass::SyncPair
        );
        assert_eq!(
            PtpMessageClass::from_first_byte(0x08),
            PtpMessageClass::SyncPair
        );
    }

    #[test]
    fn test_message_class_announce() {
        assert_eq!(
            PtpMessageClass::from_first_byte(0x0B),
            PtpMessageClass::Announce
        );
    }

    #[test]
    fn test_message_class_management() {
        assert_eq!(
            PtpMessageClass::from_first_byte(0x0D),
            PtpMessageClass::Management
        );
    }

    // -----------------------------------------------------------------------
    // PtpBatchProcessor tests
    // -----------------------------------------------------------------------

    /// Builds a minimal but structurally valid PTP message buffer.
    fn make_ptp_buf(msg_type_nibble: u8, seq_id: u16, length: u16) -> Vec<u8> {
        let mut buf = vec![0u8; length as usize];
        // Byte 0: messageType (lower nibble), version (upper nibble) — NOTE:
        // The version is stored in byte 1 per our processor (lower nibble of byte 1).
        buf[0] = msg_type_nibble & 0x0F;
        buf[1] = 0x02; // version = 2
        buf[2] = (length >> 8) as u8;
        buf[3] = length as u8;
        buf[30] = (seq_id >> 8) as u8;
        buf[31] = seq_id as u8;
        buf
    }

    #[test]
    fn test_batch_processor_enqueue_and_drain() {
        let mut proc = PtpBatchProcessor::new(16);
        for i in 0u16..5 {
            proc.enqueue(RawPtpMessage::new(
                make_ptp_buf(0x0, i, 44),
                1_000_000_000 + u64::from(i) * 100_000,
            ));
        }
        assert_eq!(proc.queue_depth(), 5);
        let results = proc.process_all();
        assert_eq!(results.len(), 5);
        assert_eq!(proc.queue_depth(), 0);
        assert_eq!(proc.processed_count(), 5);
        assert_eq!(proc.dropped_count(), 0);
    }

    #[test]
    fn test_batch_processor_overflow_drops_oldest() {
        let mut proc = PtpBatchProcessor::new(4);
        for i in 0u16..6 {
            proc.enqueue(RawPtpMessage::new(make_ptp_buf(0x0, i, 44), u64::from(i)));
        }
        // Queue capacity 4; 2 oldest should have been dropped.
        assert_eq!(proc.dropped_count(), 2);
        assert_eq!(proc.queue_depth(), 4);
    }

    #[test]
    fn test_batch_processor_invalid_short_message() {
        let mut proc = PtpBatchProcessor::new(8);
        proc.enqueue(RawPtpMessage::new(vec![0u8; 10], 0));
        let results = proc.process_all();
        assert_eq!(results.len(), 1);
        assert!(!results[0].valid);
        assert!(results[0].error.is_some());
        assert_eq!(proc.processed_count(), 0);
    }

    #[test]
    fn test_batch_processor_process_up_to() {
        let mut proc = PtpBatchProcessor::new(16);
        for i in 0u16..10 {
            proc.enqueue(RawPtpMessage::new(make_ptp_buf(0x0B, i, 64), u64::from(i)));
        }
        let partial = proc.process_up_to(4);
        assert_eq!(partial.len(), 4);
        assert_eq!(proc.queue_depth(), 6);
    }

    // -----------------------------------------------------------------------
    // SyncFollowUpPairer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pairer_match_sync_followup() {
        let mut pairer = SyncFollowUpPairer::new();
        pairer.record_sync(42, 1_000_000_000);
        let pair = pairer.record_follow_up(42, 999_000_000);
        assert!(pair.is_some(), "should produce a SyncPair");
        let p = pair.expect("should have pair");
        assert_eq!(p.sequence_id, 42);
        assert_eq!(p.sync_recv_ns, 1_000_000_000);
        assert_eq!(p.precise_origin_ns, 999_000_000);
    }

    #[test]
    fn test_pairer_follow_up_without_sync_returns_none() {
        let mut pairer = SyncFollowUpPairer::new();
        let pair = pairer.record_follow_up(99, 0);
        assert!(pair.is_none(), "no matching Sync → None");
    }

    #[test]
    fn test_pairer_pending_count() {
        let mut pairer = SyncFollowUpPairer::new();
        pairer.record_sync(1, 100);
        pairer.record_sync(2, 200);
        assert_eq!(pairer.pending_count(), 2);
        let _ = pairer.record_follow_up(1, 0);
        assert_eq!(pairer.pending_count(), 1);
    }

    #[test]
    fn test_pairer_clear() {
        let mut pairer = SyncFollowUpPairer::new();
        pairer.record_sync(5, 500);
        pairer.record_sync(6, 600);
        pairer.clear();
        assert_eq!(pairer.pending_count(), 0);
        assert!(pairer.record_follow_up(5, 0).is_none());
    }
}
