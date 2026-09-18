// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Event system for render farm.

use crate::job::JobId;
use crate::worker::WorkerId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Event type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    /// Job submitted
    JobSubmitted {
        /// Job identifier
        job_id: JobId,
        /// Event timestamp
        timestamp: DateTime<Utc>,
    },
    /// Job started
    JobStarted {
        /// Job identifier
        job_id: JobId,
        /// Event timestamp
        timestamp: DateTime<Utc>,
    },
    /// Job completed
    JobCompleted {
        /// Job identifier
        job_id: JobId,
        /// Event timestamp
        timestamp: DateTime<Utc>,
    },
    /// Job failed
    JobFailed {
        /// Job identifier
        job_id: JobId,
        /// Error description
        error: String,
        /// Event timestamp
        timestamp: DateTime<Utc>,
    },
    /// Worker registered
    WorkerRegistered {
        /// Worker identifier
        worker_id: WorkerId,
        /// Event timestamp
        timestamp: DateTime<Utc>,
    },
    /// Worker offline
    WorkerOffline {
        /// Worker identifier
        worker_id: WorkerId,
        /// Event timestamp
        timestamp: DateTime<Utc>,
    },
    /// Frame completed
    FrameCompleted {
        /// Job identifier
        job_id: JobId,
        /// Frame number
        frame: u32,
        /// Event timestamp
        timestamp: DateTime<Utc>,
    },
}

/// Event handler trait
pub trait EventHandler: Send + Sync {
    /// Handle event
    fn handle(&self, event: &Event);
}

/// Event bus
pub struct EventBus {
    handlers: Vec<Box<dyn EventHandler>>,
}

impl EventBus {
    /// Create a new event bus
    #[must_use]
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Register event handler
    pub fn register<H: EventHandler + 'static>(&mut self, handler: H) {
        self.handlers.push(Box::new(handler));
    }

    /// Publish event
    pub fn publish(&self, event: Event) {
        for handler in &self.handlers {
            handler.handle(&event);
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct TestHandler {
        events: Arc<Mutex<Vec<Event>>>,
    }

    impl EventHandler for TestHandler {
        fn handle(&self, event: &Event) {
            self.events
                .lock()
                .expect("should succeed in test")
                .push(event.clone());
        }
    }

    #[test]
    fn test_event_bus() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let handler = TestHandler {
            events: events.clone(),
        };

        let mut bus = EventBus::new();
        bus.register(handler);

        let job_id = JobId::new();
        bus.publish(Event::JobSubmitted {
            job_id,
            timestamp: Utc::now(),
        });

        assert_eq!(events.lock().expect("should succeed in test").len(), 1);
    }
}
