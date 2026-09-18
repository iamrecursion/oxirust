//! Communication primitives for distributed computing
//!
//! This module provides communication abstractions for exchanging data
//! between compute nodes, including boundary conditions, synchronization
//! barriers, and message passing.

use crate::common::IntegrateFloat;
use crate::distributed::types::{
    AckStatus, BoundaryConditions, BoundaryData, ChunkId, ChunkResult, DistributedError,
    DistributedMessage, DistributedResult, JobId, NodeCapabilities, NodeId, NodeStatus, WorkChunk,
};
use scirs2_core::ndarray::Array1;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Message channel for inter-node communication
pub struct MessageChannel<F: IntegrateFloat> {
    /// Outgoing message queue
    outbox: Mutex<VecDeque<(NodeId, DistributedMessage<F>)>>,
    /// Incoming message queue
    inbox: Mutex<VecDeque<(NodeId, DistributedMessage<F>)>>,
    /// Message ID counter
    next_message_id: AtomicU64,
    /// Pending acknowledgments
    pending_acks: Mutex<HashMap<u64, (Instant, NodeId)>>,
    /// Acknowledgment timeout
    ack_timeout: Duration,
    /// Condition variable for inbox notifications
    inbox_cv: Condvar,
    /// Mutex for condition variable
    inbox_mutex: Mutex<()>,
}

impl<F: IntegrateFloat> MessageChannel<F> {
    /// Create a new message channel
    pub fn new(ack_timeout: Duration) -> Self {
        Self {
            outbox: Mutex::new(VecDeque::new()),
            inbox: Mutex::new(VecDeque::new()),
            next_message_id: AtomicU64::new(1),
            pending_acks: Mutex::new(HashMap::new()),
            ack_timeout,
            inbox_cv: Condvar::new(),
            inbox_mutex: Mutex::new(()),
        }
    }

    /// Generate a new message ID
    pub fn generate_message_id(&self) -> u64 {
        self.next_message_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Send a message to a node
    pub fn send(&self, target: NodeId, message: DistributedMessage<F>) -> DistributedResult<u64> {
        let message_id = self.generate_message_id();

        match self.outbox.lock() {
            Ok(mut outbox) => {
                outbox.push_back((target, message));
            }
            Err(_) => {
                return Err(DistributedError::CommunicationError(
                    "Failed to acquire outbox lock".to_string(),
                ))
            }
        }

        // Track pending acknowledgment
        match self.pending_acks.lock() {
            Ok(mut pending) => {
                pending.insert(message_id, (Instant::now(), target));
            }
            Err(_) => {
                return Err(DistributedError::CommunicationError(
                    "Failed to track acknowledgment".to_string(),
                ))
            }
        }

        Ok(message_id)
    }

    /// Receive a message (blocking with timeout)
    pub fn receive(&self, timeout: Duration) -> Option<(NodeId, DistributedMessage<F>)> {
        let deadline = Instant::now() + timeout;

        loop {
            // Try to get a message
            if let Ok(mut inbox) = self.inbox.lock() {
                if let Some(msg) = inbox.pop_front() {
                    return Some(msg);
                }
            }

            // Wait for notification or timeout
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }

            if let Ok(guard) = self.inbox_mutex.lock() {
                let _ = self.inbox_cv.wait_timeout(guard, remaining);
            }
        }
    }

    /// Receive a message (non-blocking)
    pub fn try_receive(&self) -> Option<(NodeId, DistributedMessage<F>)> {
        match self.inbox.lock() {
            Ok(mut inbox) => inbox.pop_front(),
            Err(_) => None,
        }
    }

    /// Deliver a message to the inbox (called by network layer)
    pub fn deliver(&self, source: NodeId, message: DistributedMessage<F>) -> DistributedResult<()> {
        match self.inbox.lock() {
            Ok(mut inbox) => {
                inbox.push_back((source, message));
                // Notify waiting receivers
                self.inbox_cv.notify_one();
                Ok(())
            }
            Err(_) => Err(DistributedError::CommunicationError(
                "Failed to acquire inbox lock".to_string(),
            )),
        }
    }

    /// Process acknowledgment
    pub fn process_ack(&self, message_id: u64, status: AckStatus) -> DistributedResult<()> {
        match self.pending_acks.lock() {
            Ok(mut pending) => {
                if pending.remove(&message_id).is_some() {
                    if status == AckStatus::Error {
                        return Err(DistributedError::CommunicationError(
                            "Message processing failed at remote node".to_string(),
                        ));
                    }
                    Ok(())
                } else {
                    // Unknown acknowledgment, might be a duplicate
                    Ok(())
                }
            }
            Err(_) => Err(DistributedError::CommunicationError(
                "Failed to process acknowledgment".to_string(),
            )),
        }
    }

    /// Check for timed-out messages
    pub fn check_timeouts(&self) -> Vec<(u64, NodeId)> {
        match self.pending_acks.lock() {
            Ok(mut pending) => {
                let now = Instant::now();
                let timed_out: Vec<_> = pending
                    .iter()
                    .filter(|(_, (sent_at, _))| now.duration_since(*sent_at) > self.ack_timeout)
                    .map(|(id, (_, node))| (*id, *node))
                    .collect();

                for (id, _) in &timed_out {
                    pending.remove(id);
                }

                timed_out
            }
            Err(_) => Vec::new(),
        }
    }

    /// Get outbox size
    pub fn outbox_size(&self) -> usize {
        self.outbox.lock().map(|o| o.len()).unwrap_or(0)
    }

    /// Get inbox size
    pub fn inbox_size(&self) -> usize {
        self.inbox.lock().map(|i| i.len()).unwrap_or(0)
    }

    /// Drain outbox for sending
    pub fn drain_outbox(&self) -> Vec<(NodeId, DistributedMessage<F>)> {
        match self.outbox.lock() {
            Ok(mut outbox) => outbox.drain(..).collect(),
            Err(_) => Vec::new(),
        }
    }
}

/// Boundary condition exchanger for sharing data between adjacent chunks
pub struct BoundaryExchanger<F: IntegrateFloat> {
    /// Boundary data from neighbors, keyed by (target_chunk, source_chunk)
    received_boundaries: RwLock<HashMap<(ChunkId, ChunkId), BoundaryData<F>>>,
    /// Pending boundary requests
    pending_requests: Mutex<HashMap<(ChunkId, ChunkId), Instant>>,
    /// Request timeout
    timeout: Duration,
}

impl<F: IntegrateFloat> BoundaryExchanger<F> {
    /// Create a new boundary exchanger
    pub fn new(timeout: Duration) -> Self {
        Self {
            received_boundaries: RwLock::new(HashMap::new()),
            pending_requests: Mutex::new(HashMap::new()),
            timeout,
        }
    }

    /// Request boundary data from a neighbor chunk
    pub fn request_boundary(
        &self,
        target_chunk: ChunkId,
        source_chunk: ChunkId,
    ) -> DistributedResult<()> {
        match self.pending_requests.lock() {
            Ok(mut pending) => {
                pending.insert((target_chunk, source_chunk), Instant::now());
                Ok(())
            }
            Err(_) => Err(DistributedError::CommunicationError(
                "Failed to register boundary request".to_string(),
            )),
        }
    }

    /// Receive boundary data from a neighbor
    pub fn receive_boundary(
        &self,
        target_chunk: ChunkId,
        source_chunk: ChunkId,
        data: BoundaryData<F>,
    ) -> DistributedResult<()> {
        match self.received_boundaries.write() {
            Ok(mut boundaries) => {
                boundaries.insert((target_chunk, source_chunk), data);

                // Remove from pending requests
                if let Ok(mut pending) = self.pending_requests.lock() {
                    pending.remove(&(target_chunk, source_chunk));
                }

                Ok(())
            }
            Err(_) => Err(DistributedError::CommunicationError(
                "Failed to store boundary data".to_string(),
            )),
        }
    }

    /// Get boundary data for a chunk
    pub fn get_boundary(
        &self,
        target_chunk: ChunkId,
        source_chunk: ChunkId,
    ) -> Option<BoundaryData<F>> {
        match self.received_boundaries.read() {
            Ok(boundaries) => boundaries.get(&(target_chunk, source_chunk)).cloned(),
            Err(_) => None,
        }
    }

    /// Build complete boundary conditions for a chunk
    pub fn build_boundary_conditions(
        &self,
        chunk_id: ChunkId,
        left_neighbor: Option<ChunkId>,
        right_neighbor: Option<ChunkId>,
    ) -> BoundaryConditions<F> {
        let mut bc = BoundaryConditions::default();

        if let Some(left_id) = left_neighbor {
            bc.left_boundary = self.get_boundary(chunk_id, left_id);
        }

        if let Some(right_id) = right_neighbor {
            bc.right_boundary = self.get_boundary(chunk_id, right_id);
        }

        bc
    }

    /// Check for timed-out requests
    pub fn check_timeouts(&self) -> Vec<(ChunkId, ChunkId)> {
        match self.pending_requests.lock() {
            Ok(mut pending) => {
                let now = Instant::now();
                let timed_out: Vec<_> = pending
                    .iter()
                    .filter(|(_, sent_at)| now.duration_since(**sent_at) > self.timeout)
                    .map(|(key, _)| *key)
                    .collect();

                for key in &timed_out {
                    pending.remove(key);
                }

                timed_out
            }
            Err(_) => Vec::new(),
        }
    }

    /// Clear all received boundaries
    pub fn clear(&self) {
        if let Ok(mut boundaries) = self.received_boundaries.write() {
            boundaries.clear();
        }
        if let Ok(mut pending) = self.pending_requests.lock() {
            pending.clear();
        }
    }
}

/// Synchronization barrier for coordinating nodes
pub struct SyncBarrier {
    /// Barrier ID
    barrier_id: AtomicU64,
    /// Expected number of participants
    expected_count: usize,
    /// Current barrier state
    state: Mutex<BarrierState>,
    /// Condition variable for waiting
    cv: Condvar,
}

/// Internal state of a barrier
struct BarrierState {
    /// Current barrier ID being waited on
    current_id: u64,
    /// Nodes that have arrived at the barrier
    arrived: Vec<NodeId>,
    /// Whether the barrier has been released
    released: bool,
}

impl SyncBarrier {
    /// Create a new synchronization barrier
    pub fn new(expected_count: usize) -> Self {
        Self {
            barrier_id: AtomicU64::new(1),
            expected_count,
            state: Mutex::new(BarrierState {
                current_id: 1,
                arrived: Vec::new(),
                released: false,
            }),
            cv: Condvar::new(),
        }
    }

    /// Start a new barrier phase
    pub fn new_barrier(&self) -> u64 {
        let new_id = self.barrier_id.fetch_add(1, Ordering::SeqCst);

        if let Ok(mut state) = self.state.lock() {
            state.current_id = new_id;
            state.arrived.clear();
            state.released = false;
        }

        new_id
    }

    /// Signal arrival at the barrier
    pub fn arrive(&self, barrier_id: u64, node_id: NodeId) -> DistributedResult<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| DistributedError::SyncError("Failed to acquire barrier lock".into()))?;

        if state.current_id != barrier_id {
            return Err(DistributedError::SyncError(format!(
                "Barrier ID mismatch: expected {}, got {}",
                state.current_id, barrier_id
            )));
        }

        if !state.arrived.contains(&node_id) {
            state.arrived.push(node_id);
        }

        // Check if all nodes have arrived
        if state.arrived.len() >= self.expected_count {
            state.released = true;
            self.cv.notify_all();
        }

        Ok(())
    }

    /// Wait for all nodes to arrive at the barrier
    pub fn wait(&self, barrier_id: u64, timeout: Duration) -> DistributedResult<()> {
        let deadline = Instant::now() + timeout;

        let mut state = self
            .state
            .lock()
            .map_err(|_| DistributedError::SyncError("Failed to acquire barrier lock".into()))?;

        while !state.released && state.current_id == barrier_id {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(DistributedError::SyncError(
                    "Barrier wait timeout".to_string(),
                ));
            }

            let (new_state, result) = self.cv.wait_timeout(state, remaining).map_err(|_| {
                DistributedError::SyncError("Failed to wait on barrier".to_string())
            })?;

            state = new_state;

            if result.timed_out() && !state.released {
                return Err(DistributedError::SyncError(
                    "Barrier wait timeout".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Check if barrier is complete (non-blocking)
    pub fn is_complete(&self, barrier_id: u64) -> bool {
        match self.state.lock() {
            Ok(state) => state.current_id == barrier_id && state.released,
            Err(_) => false,
        }
    }

    /// Get number of arrived nodes
    pub fn arrived_count(&self) -> usize {
        match self.state.lock() {
            Ok(state) => state.arrived.len(),
            Err(_) => 0,
        }
    }
}

/// Communicator for all-to-all and collective operations
pub struct Communicator<F: IntegrateFloat> {
    /// Local node ID
    local_node_id: NodeId,
    /// Message channel
    channel: Arc<MessageChannel<F>>,
    /// Boundary exchanger
    boundary_exchanger: Arc<BoundaryExchanger<F>>,
    /// Synchronization barriers
    barriers: RwLock<HashMap<u64, Arc<SyncBarrier>>>,
    /// Known peer nodes
    peers: RwLock<Vec<NodeId>>,
}

impl<F: IntegrateFloat> Communicator<F> {
    /// Create a new communicator
    pub fn new(
        local_node_id: NodeId,
        channel: Arc<MessageChannel<F>>,
        boundary_exchanger: Arc<BoundaryExchanger<F>>,
    ) -> Self {
        Self {
            local_node_id,
            channel,
            boundary_exchanger,
            barriers: RwLock::new(HashMap::new()),
            peers: RwLock::new(Vec::new()),
        }
    }

    /// Get local node ID
    pub fn local_id(&self) -> NodeId {
        self.local_node_id
    }

    /// Add a peer node
    pub fn add_peer(&self, node_id: NodeId) -> DistributedResult<()> {
        match self.peers.write() {
            Ok(mut peers) => {
                if !peers.contains(&node_id) {
                    peers.push(node_id);
                }
                Ok(())
            }
            Err(_) => Err(DistributedError::CommunicationError(
                "Failed to add peer".to_string(),
            )),
        }
    }

    /// Remove a peer node
    pub fn remove_peer(&self, node_id: NodeId) -> DistributedResult<()> {
        match self.peers.write() {
            Ok(mut peers) => {
                peers.retain(|&id| id != node_id);
                Ok(())
            }
            Err(_) => Err(DistributedError::CommunicationError(
                "Failed to remove peer".to_string(),
            )),
        }
    }

    /// Get list of peer nodes
    pub fn get_peers(&self) -> Vec<NodeId> {
        match self.peers.read() {
            Ok(peers) => peers.clone(),
            Err(_) => Vec::new(),
        }
    }

    /// Send work chunk to a node
    pub fn send_work(
        &self,
        target: NodeId,
        chunk: WorkChunk<F>,
        deadline: Option<Duration>,
    ) -> DistributedResult<u64> {
        let message = DistributedMessage::WorkAssignment { chunk, deadline };
        self.channel.send(target, message)
    }

    /// Send chunk result back to coordinator
    pub fn send_result(&self, target: NodeId, result: ChunkResult<F>) -> DistributedResult<u64> {
        let message = DistributedMessage::WorkResult { result };
        self.channel.send(target, message)
    }

    /// Send boundary data to a neighbor
    pub fn send_boundary(
        &self,
        target: NodeId,
        source_chunk: ChunkId,
        target_chunk: ChunkId,
        boundary_data: BoundaryData<F>,
    ) -> DistributedResult<u64> {
        let message = DistributedMessage::BoundaryExchange {
            source_chunk,
            target_chunk,
            boundary_data,
        };
        self.channel.send(target, message)
    }

    /// Broadcast a message to all peers
    pub fn broadcast(&self, message: DistributedMessage<F>) -> DistributedResult<Vec<u64>> {
        let peers = self.get_peers();
        let mut message_ids = Vec::with_capacity(peers.len());

        for peer in peers {
            let id = self.channel.send(peer, message.clone())?;
            message_ids.push(id);
        }

        Ok(message_ids)
    }

    /// Create a new synchronization barrier
    pub fn create_barrier(&self, expected_count: usize) -> DistributedResult<u64> {
        let barrier = Arc::new(SyncBarrier::new(expected_count));
        let barrier_id = barrier.new_barrier();

        match self.barriers.write() {
            Ok(mut barriers) => {
                barriers.insert(barrier_id, barrier);
                Ok(barrier_id)
            }
            Err(_) => Err(DistributedError::SyncError(
                "Failed to create barrier".to_string(),
            )),
        }
    }

    /// Synchronize at a barrier
    pub fn barrier(&self, barrier_id: u64, timeout: Duration) -> DistributedResult<()> {
        // Get the barrier
        let barrier = {
            match self.barriers.read() {
                Ok(barriers) => barriers.get(&barrier_id).cloned(),
                Err(_) => None,
            }
        };

        let barrier = barrier.ok_or_else(|| {
            DistributedError::SyncError(format!("Barrier {} not found", barrier_id))
        })?;

        // Notify arrival at the barrier
        barrier.arrive(barrier_id, self.local_node_id)?;

        // Broadcast barrier arrival to peers
        let message = DistributedMessage::SyncBarrier {
            barrier_id,
            node_id: self.local_node_id,
        };
        let _ = self.broadcast(message);

        // Wait for barrier completion
        barrier.wait(barrier_id, timeout)
    }

    /// Process incoming barrier messages
    pub fn process_barrier_message(
        &self,
        barrier_id: u64,
        node_id: NodeId,
    ) -> DistributedResult<()> {
        match self.barriers.read() {
            Ok(barriers) => {
                if let Some(barrier) = barriers.get(&barrier_id) {
                    barrier.arrive(barrier_id, node_id)?;
                }
                Ok(())
            }
            Err(_) => Err(DistributedError::SyncError(
                "Failed to process barrier message".to_string(),
            )),
        }
    }

    /// Receive boundary data
    pub fn receive_boundary(
        &self,
        target_chunk: ChunkId,
        source_chunk: ChunkId,
        data: BoundaryData<F>,
    ) -> DistributedResult<()> {
        self.boundary_exchanger
            .receive_boundary(target_chunk, source_chunk, data)
    }

    /// Get boundary conditions for a chunk
    pub fn get_boundary_conditions(
        &self,
        chunk_id: ChunkId,
        left_neighbor: Option<ChunkId>,
        right_neighbor: Option<ChunkId>,
    ) -> BoundaryConditions<F> {
        self.boundary_exchanger
            .build_boundary_conditions(chunk_id, left_neighbor, right_neighbor)
    }
}

/// Helper function to serialize boundary data for transmission
pub fn serialize_boundary_data<F: IntegrateFloat>(data: &BoundaryData<F>) -> Vec<u8> {
    // Simplified serialization - in production, use a proper serialization library
    let mut bytes = Vec::new();

    // Write time as f64 bytes
    let time_f64 = data.time.to_f64().unwrap_or(0.0);
    bytes.extend_from_slice(&time_f64.to_le_bytes());

    // Write state length and values
    let state_len = data.state.len() as u64;
    bytes.extend_from_slice(&state_len.to_le_bytes());
    for val in data.state.iter() {
        let val_f64 = val.to_f64().unwrap_or(0.0);
        bytes.extend_from_slice(&val_f64.to_le_bytes());
    }

    // Write source chunk ID
    bytes.extend_from_slice(&data.source_chunk.0.to_le_bytes());

    bytes
}

/// Helper function to deserialize boundary data
pub fn deserialize_boundary_data<F: IntegrateFloat>(
    bytes: &[u8],
) -> DistributedResult<BoundaryData<F>> {
    if bytes.len() < 16 {
        return Err(DistributedError::CommunicationError(
            "Insufficient data for boundary deserialization".to_string(),
        ));
    }

    let mut offset = 0;

    // Read time
    let time_bytes: [u8; 8] = bytes[offset..offset + 8]
        .try_into()
        .map_err(|_| DistributedError::CommunicationError("Invalid time bytes".to_string()))?;
    let time_f64 = f64::from_le_bytes(time_bytes);
    let time = F::from(time_f64).ok_or_else(|| {
        DistributedError::CommunicationError("Failed to convert time".to_string())
    })?;
    offset += 8;

    // Read state length
    let len_bytes: [u8; 8] = bytes[offset..offset + 8]
        .try_into()
        .map_err(|_| DistributedError::CommunicationError("Invalid length bytes".to_string()))?;
    let state_len = u64::from_le_bytes(len_bytes) as usize;
    offset += 8;

    // Read state values
    if bytes.len() < offset + state_len * 8 + 8 {
        return Err(DistributedError::CommunicationError(
            "Insufficient data for state values".to_string(),
        ));
    }

    let mut state = Array1::zeros(state_len);
    for i in 0..state_len {
        let val_bytes: [u8; 8] = bytes[offset..offset + 8]
            .try_into()
            .map_err(|_| DistributedError::CommunicationError("Invalid value bytes".to_string()))?;
        let val_f64 = f64::from_le_bytes(val_bytes);
        state[i] = F::from(val_f64).ok_or_else(|| {
            DistributedError::CommunicationError("Failed to convert value".to_string())
        })?;
        offset += 8;
    }

    // Read source chunk ID
    let chunk_bytes: [u8; 8] = bytes[offset..offset + 8]
        .try_into()
        .map_err(|_| DistributedError::CommunicationError("Invalid chunk ID bytes".to_string()))?;
    let source_chunk = ChunkId(u64::from_le_bytes(chunk_bytes));

    Ok(BoundaryData {
        time,
        state,
        derivative: None,
        source_chunk,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_channel() {
        let channel: MessageChannel<f64> = MessageChannel::new(Duration::from_secs(5));

        let node_id = NodeId::new(1);
        let message = DistributedMessage::Heartbeat {
            node_id,
            status: NodeStatus::Available,
            timestamp: 12345,
        };

        let msg_id = channel.send(node_id, message.clone());
        assert!(msg_id.is_ok());

        // Deliver message
        channel.deliver(node_id, message).expect("Delivery failed");

        // Receive message
        let received = channel.try_receive();
        assert!(received.is_some());
    }

    #[test]
    fn test_boundary_exchanger() {
        let exchanger: BoundaryExchanger<f64> = BoundaryExchanger::new(Duration::from_secs(5));

        let target = ChunkId::new(1);
        let source = ChunkId::new(0);

        let data = BoundaryData {
            time: 1.0,
            state: Array1::from_vec(vec![1.0, 2.0, 3.0]),
            derivative: None,
            source_chunk: source,
        };

        exchanger
            .receive_boundary(target, source, data)
            .expect("Failed to receive boundary");

        let retrieved = exchanger.get_boundary(target, source);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.map(|b| b.time), Some(1.0));
    }

    #[test]
    fn test_sync_barrier() {
        let barrier = SyncBarrier::new(2);
        let barrier_id = barrier.new_barrier();

        // First node arrives
        barrier
            .arrive(barrier_id, NodeId::new(1))
            .expect("Failed to arrive");
        assert!(!barrier.is_complete(barrier_id));

        // Second node arrives
        barrier
            .arrive(barrier_id, NodeId::new(2))
            .expect("Failed to arrive");
        assert!(barrier.is_complete(barrier_id));
    }

    #[test]
    fn test_boundary_serialization() {
        let data = BoundaryData {
            time: 1.5,
            state: Array1::from_vec(vec![1.0, 2.0, 3.0]),
            derivative: None,
            source_chunk: ChunkId::new(42),
        };

        let bytes = serialize_boundary_data(&data);
        let deserialized: BoundaryData<f64> =
            deserialize_boundary_data(&bytes).expect("Deserialization failed");

        assert!((deserialized.time - data.time).abs() < 1e-10);
        assert_eq!(deserialized.state.len(), data.state.len());
        assert_eq!(deserialized.source_chunk.0, data.source_chunk.0);
    }
}
