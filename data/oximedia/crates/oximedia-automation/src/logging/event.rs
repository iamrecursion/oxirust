//! Event logging for automation system.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::SystemTime;
use tokio::sync::mpsc;
use tracing::debug;

/// Event severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventSeverity {
    /// Debug information
    Debug,
    /// Informational message
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical error
    Critical,
}

/// Automation event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event severity
    pub severity: EventSeverity,
    /// Event source (e.g., "channel_0", "master_control")
    pub source: String,
    /// Event type (e.g., "playlist_start", "failover_triggered")
    pub event_type: String,
    /// Event message
    pub message: String,
    /// Additional data
    pub data: std::collections::HashMap<String, String>,
}

impl AutomationEvent {
    /// Create a new automation event.
    pub fn new(
        severity: EventSeverity,
        source: String,
        event_type: String,
        message: String,
    ) -> Self {
        Self {
            timestamp: SystemTime::now(),
            severity,
            source,
            event_type,
            message,
            data: std::collections::HashMap::new(),
        }
    }

    /// Add data field to event.
    pub fn with_data(mut self, key: String, value: String) -> Self {
        self.data.insert(key, value);
        self
    }
}

/// Event logger.
#[allow(dead_code)]
pub struct EventLogger {
    log_path: Option<PathBuf>,
    event_tx: mpsc::UnboundedSender<AutomationEvent>,
}

impl EventLogger {
    /// Create a new event logger.
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel();

        // Spawn logging task
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                // In a real implementation, this would write to log files
                debug!("Event logged: {:?}", event);
            }
        });

        Self {
            log_path: None,
            event_tx: tx,
        }
    }

    /// Create with log file path.
    pub fn with_path(log_path: PathBuf) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel();

        let path = log_path.clone();

        // Spawn logging task
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if let Err(e) = Self::write_to_file(&path, &event) {
                    eprintln!("Failed to write event log: {e}");
                }
            }
        });

        Self {
            log_path: Some(log_path),
            event_tx: tx,
        }
    }

    /// Log an event.
    pub fn log(&self, event: AutomationEvent) {
        let _ = self.event_tx.send(event);
    }

    /// Log info event.
    pub fn info(&self, source: String, event_type: String, message: String) {
        let event = AutomationEvent::new(EventSeverity::Info, source, event_type, message);
        self.log(event);
    }

    /// Log warning event.
    pub fn warning(&self, source: String, event_type: String, message: String) {
        let event = AutomationEvent::new(EventSeverity::Warning, source, event_type, message);
        self.log(event);
    }

    /// Log error event.
    pub fn error(&self, source: String, event_type: String, message: String) {
        let event = AutomationEvent::new(EventSeverity::Error, source, event_type, message);
        self.log(event);
    }

    /// Write event to file.
    fn write_to_file(path: &PathBuf, event: &AutomationEvent) -> Result<()> {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;

        let json = serde_json::to_string(event)?;
        writeln!(file, "{json}")?;

        Ok(())
    }
}

impl Default for EventLogger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let event = AutomationEvent::new(
            EventSeverity::Info,
            "channel_0".to_string(),
            "playlist_start".to_string(),
            "Playlist started".to_string(),
        );

        assert_eq!(event.severity, EventSeverity::Info);
        assert_eq!(event.source, "channel_0");
    }

    #[test]
    fn test_event_with_data() {
        let event = AutomationEvent::new(
            EventSeverity::Info,
            "test".to_string(),
            "test".to_string(),
            "test".to_string(),
        )
        .with_data("key".to_string(), "value".to_string());

        assert_eq!(event.data.get("key"), Some(&"value".to_string()));
    }

    #[tokio::test]
    async fn test_event_logger() {
        let logger = EventLogger::new();

        logger.info(
            "test".to_string(),
            "test_event".to_string(),
            "Test message".to_string(),
        );
    }

    #[test]
    fn test_severity_ordering() {
        assert!(EventSeverity::Critical > EventSeverity::Error);
        assert!(EventSeverity::Error > EventSeverity::Warning);
        assert!(EventSeverity::Warning > EventSeverity::Info);
        assert!(EventSeverity::Info > EventSeverity::Debug);
    }
}
