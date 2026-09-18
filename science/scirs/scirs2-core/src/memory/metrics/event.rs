//! Memory event tracking system
//!
//! This module provides types for tracking memory-related events such as
//! allocations, deallocations, and accesses.

use std::collections::HashMap;
use std::fmt;
use std::thread::ThreadId;
use std::time::Instant;

/// Type of memory event
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryEventType {
    /// Memory allocation
    Allocation,
    /// Memory deallocation
    Deallocation,
    /// Memory resize
    Resize,
    /// Memory access
    Access,
    /// Memory transfer (e.g., between CPU and GPU)
    Transfer,
}

impl fmt::Display for MemoryEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryEventType::Allocation => write!(f, "Allocation"),
            MemoryEventType::Deallocation => write!(f, "Deallocation"),
            MemoryEventType::Resize => write!(f, "Resize"),
            MemoryEventType::Access => write!(f, "Access"),
            MemoryEventType::Transfer => write!(f, "Transfer"),
        }
    }
}

/// A memory event representing an allocation, deallocation, etc.
#[derive(Debug, Clone)]
pub struct MemoryEvent {
    /// Event type
    pub event_type: MemoryEventType,
    /// Component that generated the event (e.g., "BufferPool", "ChunkProcessor")
    pub component: String,
    /// Optional operation context (e.g., "matrix_multiply", "fft")
    pub context: Option<String>,
    /// Allocation size in bytes
    pub size: usize,
    /// Memory address (for correlation)
    pub address: usize,
    /// Thread ID
    pub thread_id: ThreadId,
    /// Timestamp
    pub timestamp: Instant,
    /// Call stack (if enabled)
    pub call_stack: Option<Vec<String>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl MemoryEvent {
    /// Create a new memory event
    pub fn new(
        event_type: MemoryEventType,
        component: impl Into<String>,
        size: usize,
        address: usize,
    ) -> Self {
        Self {
            event_type,
            component: component.into(),
            context: None,
            size,
            address,
            thread_id: std::thread::current().id(),
            timestamp: Instant::now(),
            call_stack: None,
            metadata: HashMap::new(),
        }
    }

    /// Add context to the event
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Add call stack to the event (if enabled)
    pub fn with_call_stack(mut self) -> Self {
        if cfg!(feature = "memory_call_stack") {
            self.call_stack = Some(capture_call_stack(3)); // Skip 3 frames
        }
        self
    }

    /// Add metadata to the event
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Capture current call stack (simplified implementation)
fn capture_call_stack(skipframes: usize) -> Vec<String> {
    // In a real implementation, this would use backtrace crate or similar
    vec![format!("frame_{}", skipframes)]
}

/// Capture a call stack (simplified implementation)
#[cfg(feature = "memory_call_stack")]
#[allow(dead_code)]
fn capture_stack_frames(maxframes: usize) -> Vec<String> {
    // This is a placeholder. In a real implementation, we would use
    // the backtrace crate to capture the call stack.
    vec!["<callstack not available>".to_string()]
}

#[cfg(not(feature = "memory_call_stack"))]
#[allow(dead_code)]
fn capture_stack_frames(maxframes: usize) -> Vec<String> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_event_creation() {
        let event = MemoryEvent::new(
            MemoryEventType::Allocation,
            "TestComponent",
            1024,
            0xdeadbeef,
        );

        assert_eq!(event.event_type, MemoryEventType::Allocation);
        assert_eq!(event.component, "TestComponent");
        assert_eq!(event.size, 1024);
        assert_eq!(event.address, 0xdeadbeef);
        assert_eq!(event.context, None);
        assert!(event.call_stack.is_none());
        assert!(event.metadata.is_empty());
    }

    #[test]
    fn test_memory_event_with_context() {
        let event = MemoryEvent::new(
            MemoryEventType::Allocation,
            "TestComponent",
            1024,
            0xdeadbeef,
        )
        .with_context("TestContext");

        assert_eq!(event.context, Some("TestContext".to_string()));
    }

    #[test]
    fn test_memory_event_with_metadata() {
        let event = MemoryEvent::new(
            MemoryEventType::Allocation,
            "TestComponent",
            1024,
            0xdeadbeef,
        )
        .with_metadata("key1", "value1")
        .with_metadata("key2", "value2");

        assert_eq!(event.metadata.len(), 2);
        assert_eq!(event.metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(event.metadata.get("key2"), Some(&"value2".to_string()));
    }
}
