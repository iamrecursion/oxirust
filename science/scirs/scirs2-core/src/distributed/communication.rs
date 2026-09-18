//! Distributed communication protocols
//!
//! This module provides communication protocols for distributed computing,
//! including message passing, synchronization, and coordination mechanisms.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{mpsc, Arc, Mutex};

/// Message types for distributed communication
#[derive(Debug, Clone)]
pub enum DistributedMessage {
    /// Task assignment message
    TaskAssignment { taskid: String, payload: Vec<u8> },
    /// Result message
    Result { taskid: String, result: Vec<u8> },
    /// Heartbeat message
    Heartbeat { nodeid: String, timestamp: u64 },
    /// Coordination message
    Coordination { messagetype: String, data: Vec<u8> },
    /// Synchronization barrier
    Barrier {
        barrier_id: String,
        node_count: usize,
    },
}

/// Communication endpoint for a node
pub struct CommunicationEndpoint {
    nodeid: String,
    address: SocketAddr,
    message_handlers: Arc<Mutex<HashMap<String, Box<dyn MessageHandler + Send + Sync>>>>,
    sender: mpsc::Sender<DistributedMessage>,
    receiver: Arc<Mutex<mpsc::Receiver<DistributedMessage>>>,
}

impl CommunicationEndpoint {
    /// Create a new communication endpoint
    pub fn new(nodeid: String, address: SocketAddr) -> Self {
        let (sender, receiver) = mpsc::channel();

        Self {
            nodeid,
            address,
            message_handlers: Arc::new(Mutex::new(HashMap::new())),
            sender,
            receiver: Arc::new(Mutex::new(receiver)),
        }
    }

    /// Send a message to another node
    pub fn send_message(&self, message: DistributedMessage) -> CoreResult<()> {
        self.sender.send(message).map_err(|e| {
            CoreError::CommunicationError(ErrorContext::new(format!("Failed to send message: {e}")))
        })?;
        Ok(())
    }

    /// Register a message handler
    pub fn register_handler<H>(&self, messagetype: String, handler: H) -> CoreResult<()>
    where
        H: MessageHandler + Send + Sync + 'static,
    {
        let mut handlers = self.message_handlers.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire handlers lock".to_string(),
            ))
        })?;
        handlers.insert(messagetype, Box::new(handler));
        Ok(())
    }

    /// Process incoming messages
    pub fn process_messages(&self) -> CoreResult<()> {
        let receiver = self.receiver.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire receiver lock".to_string(),
            ))
        })?;

        while let Ok(message) = receiver.try_recv() {
            self.handle_message(message)?;
        }

        Ok(())
    }

    fn handle_message(&self, message: DistributedMessage) -> CoreResult<()> {
        let handlers = self.message_handlers.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire handlers lock".to_string(),
            ))
        })?;

        match &message {
            DistributedMessage::TaskAssignment { .. } => {
                if let Some(handler) = handlers.get("task_assignment") {
                    handler.handle(&message)?;
                }
            }
            DistributedMessage::Result { .. } => {
                if let Some(handler) = handlers.get("result") {
                    handler.handle(&message)?;
                }
            }
            DistributedMessage::Heartbeat { .. } => {
                if let Some(handler) = handlers.get("heartbeat") {
                    handler.handle(&message)?;
                }
            }
            DistributedMessage::Coordination { messagetype, .. } => {
                if let Some(handler) = handlers.get(messagetype) {
                    handler.handle(&message)?;
                }
            }
            DistributedMessage::Barrier { .. } => {
                if let Some(handler) = handlers.get("barrier") {
                    handler.handle(&message)?;
                }
            }
        }

        Ok(())
    }

    /// Get node ID
    pub fn nodeid(&self) -> &str {
        &self.nodeid
    }

    /// Get address
    pub fn address(&self) -> SocketAddr {
        self.address
    }
}

/// Trait for handling distributed messages
pub trait MessageHandler {
    /// Handle a received message
    fn handle(&self, message: &DistributedMessage) -> CoreResult<()>;
}

/// Default heartbeat handler
#[derive(Debug)]
pub struct HeartbeatHandler {
    #[allow(dead_code)]
    nodeid: String,
}

impl HeartbeatHandler {
    /// Create a new heartbeat handler
    pub fn new(nodeid: String) -> Self {
        Self { nodeid }
    }
}

impl MessageHandler for HeartbeatHandler {
    fn handle(&self, message: &DistributedMessage) -> CoreResult<()> {
        if let DistributedMessage::Heartbeat { nodeid, timestamp } = message {
            println!("Received heartbeat from {nodeid} at {timestamp}");
        }
        Ok(())
    }
}

/// Communication manager for coordinating distributed operations
pub struct CommunicationManager {
    endpoints: HashMap<String, CommunicationEndpoint>,
    local_nodeid: String,
}

impl CommunicationManager {
    /// Create a new communication manager
    pub fn new(local_nodeid: String) -> Self {
        Self {
            endpoints: HashMap::new(),
            local_nodeid,
        }
    }

    /// Add a communication endpoint
    pub fn add_endpoint(&mut self, endpoint: CommunicationEndpoint) {
        let nodeid = endpoint.nodeid().to_string();
        self.endpoints.insert(nodeid, endpoint);
    }

    /// Broadcast a message to all nodes
    pub fn broadcast(&self, message: DistributedMessage) -> CoreResult<()> {
        for endpoint in self.endpoints.values() {
            endpoint.send_message(message.clone())?;
        }
        Ok(())
    }

    /// Send a message to a specific node
    pub fn send_to(&self, nodeid: &str, message: DistributedMessage) -> CoreResult<()> {
        if let Some(endpoint) = self.endpoints.get(nodeid) {
            endpoint.send_message(message)?;
        } else {
            return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "Unknown node: {nodeid}"
            ))));
        }
        Ok(())
    }

    /// Process all pending messages
    pub fn process_all_messages(&self) -> CoreResult<()> {
        for endpoint in self.endpoints.values() {
            endpoint.process_messages()?;
        }
        Ok(())
    }

    /// Get local node ID
    pub fn local_nodeid(&self) -> &str {
        &self.local_nodeid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_communication_endpoint_creation() {
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let endpoint = CommunicationEndpoint::new("node1".to_string(), address);

        assert_eq!(endpoint.nodeid(), "node1");
        assert_eq!(endpoint.address(), address);
    }

    #[test]
    fn test_heartbeat_handler() {
        let handler = HeartbeatHandler::new("node1".to_string());
        let message = DistributedMessage::Heartbeat {
            nodeid: "node2".to_string(),
            timestamp: 123456789,
        };

        assert!(handler.handle(&message).is_ok());
    }

    #[test]
    fn test_communication_manager() {
        let mut manager = CommunicationManager::new("local_node".to_string());
        assert_eq!(manager.local_nodeid(), "local_node");

        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let endpoint = CommunicationEndpoint::new("node1".to_string(), address);
        manager.add_endpoint(endpoint);

        assert!(manager.endpoints.contains_key("node1"));
    }
}
